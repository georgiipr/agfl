"""Pre-declared analysis of the SI_Hom confirmatory experiment (plan v1).

Standalone: numpy and scipy only. It never imports the training package and
reads only saved run records (result.json, predictions.npz), so it can run on
the cluster or on a downloaded ``report/`` folder.

    python scripts/confirmatory_analysis.py results/si-hom-confirmatory-v1

The hypotheses, arms, seeds, tests and decision labels below are fixed by
docs/confirmatory.md and were written before any confirmatory run existed.
Change nothing here after the runs have been seen; a different question is a
new, exploratory analysis.

``--exploratory`` applies the same computations to other seeds or epoch budgets
(for example the first, exploratory run); its output is labelled accordingly and
is not a confirmatory result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import stats

PLAN = "si-hom-confirmatory-v1"
SEEDS = list(range(5, 25))
ALPHA = 0.05
METRIC = "accuracy"

DATA = {"classes": [0, 1, 2, 3, 4, 5, 6, 7], "cohort": "pooled", "data_dir": "../AGFL_speech_data",
        "normalization": "train_channel", "start": 0, "subjects": [1, 2, 3, 4, 5, 6, 7], "window": 500}
SPLIT = {"protocol": "stratified", "train": 0.6, "validation": 0.2, "test": 0.2}
RECIPES = {
    # The project recipe of the first run, with the declared 60-epoch budget.
    "ours": {"epochs": 60, "batch_size": 64, "optimizer": "adamw", "learning_rate": 0.001, "weight_decay": 0.001,
             "class_weights": "balanced", "scheduler": "warmup_cosine", "warmup_epochs": 10,
             "loss": "cross_entropy", "checkpoint_criterion": "loss"},
    # The dataset authors' training settings (their MyEEGNet/train.py).
    "authors": {"epochs": 50, "batch_size": 16, "optimizer": "adam", "learning_rate": 0.001, "weight_decay": 0.0,
                "class_weights": "none", "scheduler": "none", "warmup_epochs": 0,
                "loss": "cross_entropy", "checkpoint_criterion": "loss"},
}
MODEL_OPTIONS = {
    "eegnet": {"electrode_architecture": "pre_spatial", "temp_kernel": 250},
    "eegnet_original": {"electrode_architecture": "pre_spatial", "temp_kernel": 32, "max_norm_schedule": "init"},
}
# arm name -> (backbone, attention, recipe)
ARMS = {
    "eegnet+agfl": ("eegnet", "agfl", "ours"),
    "eegnet+mha": ("eegnet", "mha", "ours"),
    "eegnet+hcann": ("eegnet", "hcann", "ours"),
    "original+agfl": ("eegnet_original", "agfl", "ours"),
    "original+mha": ("eegnet_original", "mha", "ours"),
    "original+hcann": ("eegnet_original", "hcann", "ours"),
    "original+mha@authors": ("eegnet_original", "mha", "authors"),
    "original+hcann@authors": ("eegnet_original", "hcann", "authors"),
}
# Contrasts are weighted sums of per-seed test accuracies; positive favours "ours".
PRIMARY = [
    ("H1", "Our EEGNet minus the authors' EEGNet settings, averaged over the three attentions",
     {"eegnet+agfl": 1 / 3, "eegnet+mha": 1 / 3, "eegnet+hcann": 1 / 3,
      "original+agfl": -1 / 3, "original+mha": -1 / 3, "original+hcann": -1 / 3}),
    ("H2", "AGFL minus the authors' attention (HCANN), averaged over the two EEGNets",
     {"eegnet+agfl": 0.5, "original+agfl": 0.5, "eegnet+hcann": -0.5, "original+hcann": -0.5}),
    ("H3", "MHA minus the authors' attention (HCANN), averaged over the two EEGNets",
     {"eegnet+mha": 0.5, "original+mha": 0.5, "eegnet+hcann": -0.5, "original+hcann": -0.5}),
]
SECONDARY = [
    ("S1", "Our EEGNet with AGFL minus the authors' settings with their attention",
     {"eegnet+agfl": 1, "original+hcann": -1}),
    ("S2", "Our EEGNet with AGFL minus the authors' settings with MHA under the authors' training recipe",
     {"eegnet+agfl": 1, "original+mha@authors": -1}),
    ("S3", "Our EEGNet with AGFL minus the authors' settings with their attention under the authors' training recipe",
     {"eegnet+agfl": 1, "original+hcann@authors": -1}),
    ("S4", "AGFL minus MHA, averaged over the two EEGNets",
     {"eegnet+agfl": 0.5, "original+agfl": 0.5, "eegnet+mha": -0.5, "original+mha": -0.5}),
    ("S5", "Project training recipe minus the authors' recipe on the authors' EEGNet settings (MHA and HCANN)",
     {"original+mha": 0.5, "original+hcann": 0.5, "original+mha@authors": -0.5, "original+hcann@authors": -0.5}),
]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_results(root):
    """Completed run records, preferring the report mirror over artifacts."""
    root = Path(root).expanduser().resolve()
    for candidate in (root / "report" / "runs", root / "runs", root / "artifacts", root):
        if candidate.is_dir():
            paths = sorted(candidate.rglob("result.json"))
            if paths:
                return paths
    return []


def classify(config, recipes):
    """Name of the declared arm a saved configuration belongs to, or the reason it is not one."""
    if config.get("dataset") != "si_hom" or config.get("data") != DATA:
        return None, "data settings differ from the plan"
    if config.get("split") != SPLIT:
        return None, "split settings differ from the plan"
    if config.get("subject_id") is not None:
        return None, "not a pooled-cohort run"
    model, attention = config.get("model"), config.get("attention")
    expected = MODEL_OPTIONS.get(model)
    if expected is None or any(config["model_options"].get(key) != value for key, value in expected.items()):
        return None, "backbone or its options are not part of the plan"
    if config["attention_options"].get("heads") != 4:
        return None, "attention head count differs from the plan"
    training = config["training"]
    for recipe, settings in recipes.items():
        if all(training.get(key) == value for key, value in settings.items()):
            for name, arm in ARMS.items():
                if arm == (model, attention, recipe):
                    return name, None
    return None, "training recipe or attention is not part of the plan"


def load_runs(root, recipes, seeds):
    runs, ignored = {}, []
    for path in find_results(root):
        if path.with_name("failure.json").exists():
            ignored.append((str(path), "failed run"))
            continue
        record = json.loads(path.read_text())
        if record.get("status") != "completed":
            ignored.append((str(path), "not completed"))
            continue
        arm, reason = classify(record["config"], recipes)
        if arm is None:
            ignored.append((str(path), reason))
            continue
        if record["seed"] not in seeds:
            ignored.append((str(path), f"seed {record['seed']} is not a declared seed"))
            continue
        key = (arm, record["seed"])
        if key in runs:
            raise ValueError(f"Two completed runs for {arm} seed {record['seed']}: {runs[key]['path']} and {path}")
        predictions = path.with_name("predictions.npz")
        history = path.with_name("history.json")
        runs[key] = {
            "path": str(path), "arm": arm, "seed": record["seed"],
            "test": record["test"], "validation": record["validation"],
            "split_id": record["split_id"], "fingerprint": record["dataset_fingerprint"],
            "source_sha256": record["config"].get("provenance", {}).get("source_sha256"),
            "best_epoch": record["best_checkpoint_epoch"], "epochs": record["epochs_trained"],
            "n_train": int(sum(record["training_class_counts"])), "n_test": int(record["test"]["n_samples"]),
            "elapsed": record.get("elapsed_seconds"), "predictions": predictions if predictions.is_file() else None,
            "history": history if history.is_file() else None,
        }
    return runs, ignored


def check_consistency(runs):
    """All arms must share data, code and, within a seed, the split."""
    problems = []
    if len({run["fingerprint"] for run in runs.values()}) > 1:
        problems.append("runs were trained on different dataset fingerprints")
    if len({run["source_sha256"] for run in runs.values()}) > 1:
        problems.append("runs were trained with different source code")
    if len({(run["n_train"], run["n_test"]) for run in runs.values()}) > 1:
        problems.append("training or test partition sizes differ between runs")
    for seed in sorted({seed for _, seed in runs}):
        if len({run["split_id"] for (_, s), run in runs.items() if s == seed}) > 1:
            problems.append(f"seed {seed}: arms did not use the same split")
    return problems


def holm(pvalues):
    adjusted = [None] * len(pvalues)
    order = sorted((p, i) for i, p in enumerate(pvalues) if p is not None)
    running = 0.0
    for rank, (p, index) in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p))
        adjusted[index] = running
    return adjusted


def seed_statistics(differences, ratio):
    """Paired tests with the seed (one split and one initialisation) as the unit.

    ``corrected`` is the Nadeau-Bengio resampled t-test: random train/test
    splits of one finite dataset overlap, so the variance of the mean difference
    is inflated by 1/n + n_test/n_train instead of 1/n. ``naive`` is the plain
    paired t-test, valid only for the expected difference over splits and
    initialisations of this particular dataset.
    """
    d = np.asarray(differences, dtype=float)
    n = len(d)
    out = {"n_seeds": n, "mean": float(d.mean()) if n else None, "positive_seeds": int((d > 0).sum()),
           "negative_seeds": int((d < 0).sum())}
    if n < 2:
        return {**out, "sd": None, "corrected_p": None, "naive_p": None, "wilcoxon_p": None,
                "corrected_ci": None, "dz": None}
    sd = float(d.std(ddof=1))
    out["sd"] = sd
    if sd == 0:
        return {**out, "corrected_p": None, "naive_p": None, "wilcoxon_p": None, "corrected_ci": None, "dz": None}
    quantile = float(stats.t.ppf(1 - ALPHA / 2, n - 1))
    naive_se = sd / math.sqrt(n)
    corrected_se = sd * math.sqrt(1 / n + ratio)
    nonzero = d[d != 0]
    out.update(
        dz=out["mean"] / sd,
        naive_t=out["mean"] / naive_se, naive_p=float(2 * stats.t.sf(abs(out["mean"] / naive_se), n - 1)),
        naive_ci=[out["mean"] - quantile * naive_se, out["mean"] + quantile * naive_se],
        corrected_t=out["mean"] / corrected_se,
        corrected_p=float(2 * stats.t.sf(abs(out["mean"] / corrected_se), n - 1)),
        corrected_ci=[out["mean"] - quantile * corrected_se, out["mean"] + quantile * corrected_se],
        wilcoxon_p=float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) else None,
    )
    return out


def trial_statistics(runs, weights, seeds):
    """Supporting view: the unique test trial as the unit, the trained models taken as given."""
    per_trial = {}
    for seed in seeds:
        reference = None
        for arm, weight in weights.items():
            run = runs[(arm, seed)]
            if run["predictions"] is None:
                return None
            with np.load(run["predictions"], allow_pickle=False) as saved:
                ids = saved["test_ids"].astype(str)
                correct = saved["test_probabilities"].argmax(1) == saved["test_targets"]
            if reference is None:
                reference, values = ids, np.zeros(len(ids))
            elif not np.array_equal(reference, ids):
                raise ValueError(f"seed {seed}: arms hold different test trials")
            values = values + weight * correct
        for identifier, value in zip(reference, values):
            per_trial.setdefault(identifier, []).append(float(value))
    d = np.asarray([np.mean(values) for values in per_trial.values()])
    if len(d) < 2 or d.std(ddof=1) == 0:
        return None
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    return {"unique_trials": len(d), "mean": float(d.mean()), "ci": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
            "p": float(2 * stats.t.sf(abs(d.mean() / se), len(d) - 1))}


def evaluate(runs, family, seeds, ratio, exploratory):
    rows = []
    for name, description, weights in family:
        common = [seed for seed in seeds if all((arm, seed) in runs for arm in weights)]
        complete = common == list(seeds)
        row = {"id": name, "description": description, "arms": weights, "seeds_used": common, "complete": complete}
        for partition in ("test", "validation"):
            differences = [sum(weight * runs[(arm, seed)][partition][METRIC] for arm, weight in weights.items())
                           for seed in common]
            row[partition] = seed_statistics(differences, ratio) if common else None
            row[partition + "_per_seed"] = differences
        row["trial_level"] = trial_statistics(runs, weights, common) if common else None
        rows.append(row)
    for kind in ("corrected", "naive"):
        adjusted = holm([row["test"][kind + "_p"] if row["test"] else None for row in rows])
        for row, value in zip(rows, adjusted):
            row[kind + "_p_holm"] = value
    for row in rows:
        row["verdict"] = verdict(row, exploratory)
    return rows


def verdict(row, exploratory):
    test = row["test"]
    if not test or test.get("corrected_p") is None:
        return "NOT EVALUATED: runs are missing or all differences are identical"
    if not row["complete"] and not exploratory:
        return f"INCOMPLETE: {len(row['seeds_used'])} of {len(SEEDS)} declared seeds; no confirmatory statement"
    direction = "in favour of ours" if test["mean"] > 0 else "against ours"
    if row["corrected_p_holm"] is not None and row["corrected_p_holm"] < ALPHA:
        label = f"CONFIRMED {direction}" if test["mean"] > 0 else f"CONTRADICTED ({direction})"
    elif row["naive_p_holm"] is not None and row["naive_p_holm"] < ALPHA:
        label = (f"DIFFERENCE ON THIS DATASET ONLY, {direction}: consistent over splits and initialisations "
                 "of these recordings, not established beyond them")
    else:
        label = "NOT ESTABLISHED: no reliable difference"
    validation = row["validation"]
    if validation and validation["mean"] is not None and test["mean"] * validation["mean"] <= 0:
        label += "; NOT ROBUST (the validation partition shows the opposite sign or zero)"
    return ("EXPLORATORY, " if exploratory else "") + label


def points(value):
    return "n/a" if value is None else f"{100 * value:+.2f}"


def pvalue(value):
    return "n/a" if value is None else ("<0.001" if value < 0.001 else f"{value:.3f}")


def arm_summary(runs, seeds):
    rows = []
    for arm in ARMS:
        present = [runs[(arm, seed)] for seed in seeds if (arm, seed) in runs]
        if not present:
            rows.append({"arm": arm, "n_seeds": 0})
            continue
        test = np.array([run["test"][METRIC] for run in present])
        validation = np.array([run["validation"][METRIC] for run in present])
        rows.append({
            "arm": arm, "n_seeds": len(present), "missing_seeds": [s for s in seeds if (arm, s) not in runs],
            "test_mean": float(test.mean()), "test_sd": float(test.std(ddof=1)) if len(test) > 1 else None,
            "validation_mean": float(validation.mean()),
            "best_epoch_mean": float(np.mean([run["best_epoch"] for run in present])),
            # A checkpoint chosen at the final epoch suggests the budget was too short.
            "selected_at_last_epoch": int(sum(run["best_epoch"] == run["epochs"] for run in present)),
            "seconds_mean": float(np.mean([run["elapsed"] for run in present if run["elapsed"] is not None] or [float("nan")])),
        })
    return rows


def render(result):
    lines = [f"# SI_Hom confirmatory analysis ({result['plan']})", ""]
    lines += [f"**Status: {result['status']}**", ""]
    if result["problems"]:
        lines += ["Consistency problems (no inference should be drawn until they are resolved):", ""]
        lines += [f"- {problem}" for problem in result["problems"]] + [""]
    lines += [f"Seeds: {result['seeds'][0]}–{result['seeds'][-1]} ({len(result['seeds'])}). "
              f"Outcome: test accuracy, 8 classes, chance 12.5%. Partition sizes: train {result['n_train']}, "
              f"test {result['n_test']} (variance correction 1/n + {result['ratio']:.3f}).", "",
              "## Arms", "",
              "| Arm | Seeds | Test accuracy, mean ± SD | Validation accuracy | Mean selected epoch | Selected at last epoch | Seconds per fit |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["arms"]:
        if not row["n_seeds"]:
            lines.append(f"| {row['arm']} | 0 | missing | | | | |")
            continue
        sd = "n/a" if row["test_sd"] is None else f"{100 * row['test_sd']:.2f}"
        lines.append(f"| {row['arm']} | {row['n_seeds']} | {100 * row['test_mean']:.2f} ± {sd} | "
                     f"{100 * row['validation_mean']:.2f} | {row['best_epoch_mean']:.1f} | "
                     f"{row['selected_at_last_epoch']} | {row['seconds_mean']:.0f} |")
    for title, key, note in (
            ("Primary hypotheses", "primary",
             "Holm correction over these three tests. The corrected test decides; the plain paired test describes "
             "this dataset only."),
            ("Secondary contrasts", "secondary", "Holm correction within this family. Not decisive on their own.")):
        lines += ["", f"## {title}", "", note, "",
                  "| | Contrast | Seeds | Mean difference (points) | 95% CI, corrected | Seeds + / − | Corrected p (Holm) | Plain paired p (Holm) | Wilcoxon p | Validation difference | Per-trial difference [95% CI] |",
                  "|---|---|---:|---:|---|---|---|---|---|---:|---|"]
        for row in result[key]:
            test, validation, trial = row["test"], row["validation"], row["trial_level"]
            if not test or test["mean"] is None:
                lines.append(f"| {row['id']} | {row['description']} | 0 | missing | | | | | | | |")
                continue
            interval = "n/a" if not test.get("corrected_ci") else f"[{points(test['corrected_ci'][0])}, {points(test['corrected_ci'][1])}]"
            trial_text = "n/a" if not trial else f"{points(trial['mean'])} [{points(trial['ci'][0])}, {points(trial['ci'][1])}]"
            lines.append(
                f"| {row['id']} | {row['description']} | {test['n_seeds']} | {points(test['mean'])} | {interval} | "
                f"{test['positive_seeds']} / {test['negative_seeds']} | {pvalue(test.get('corrected_p'))} "
                f"({pvalue(row['corrected_p_holm'])}) | {pvalue(test.get('naive_p'))} ({pvalue(row['naive_p_holm'])}) | "
                f"{pvalue(test.get('wilcoxon_p'))} | {points(validation['mean']) if validation else 'n/a'} | {trial_text} |")
        lines += [""] + [f"- **{row['id']}: {row['verdict']}**" for row in result[key]]
    lines += ["", "## How to read this", "",
              "- A positive difference favours our side of the contrast (our EEGNet, AGFL or MHA).",
              "- *Corrected p* uses the Nadeau–Bengio resampled t-test with the seed as the unit. It accounts for "
              "the overlap between random splits of one finite dataset and is the pre-declared decision test.",
              "- *Plain paired p* treats seeds as independent. It answers only whether the difference is consistent "
              "over splits and initialisations of these 2640 trials.",
              "- *Validation difference* is the same contrast on the validation partition (a robustness check; that "
              "partition also selects checkpoints, identically for every arm).",
              "- *Per-trial difference* takes each unique test trial as the unit and the trained models as given; "
              "it is supporting evidence, not a decision test.", ""]
    if result["ignored"]:
        lines += [f"{len(result['ignored'])} saved runs were not part of the plan and were ignored "
                  "(listed in confirmatory.json).", ""]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="Session folder, its report/ folder, or a downloaded report")
    parser.add_argument("--output", help="Folder for confirmatory.md/.json/.csv; default <results>/confirmatory "
                                         "(inside report/ when the session folder is given)")
    parser.add_argument("--exploratory", action="store_true",
                        help="Apply the same computations outside the plan; the output is labelled exploratory")
    parser.add_argument("--seeds", type=int, nargs="+", help="Exploratory only: seeds to use")
    parser.add_argument("--epochs", type=int, help="Exploratory only: epoch budget of the project-recipe arms")
    args = parser.parse_args(argv)
    if (args.seeds or args.epochs) and not args.exploratory:
        parser.error("--seeds and --epochs change the plan; they require --exploratory")
    seeds = sorted(args.seeds) if args.seeds else list(SEEDS)
    recipes = {name: dict(settings) for name, settings in RECIPES.items()}
    if args.epochs:
        recipes["ours"]["epochs"] = args.epochs
    root = Path(args.results).expanduser().resolve()
    runs, ignored = load_runs(root, recipes, seeds)
    if not runs:
        raise SystemExit(f"No completed runs of the declared arms and seeds under {root}")
    problems = check_consistency(runs)
    first = next(iter(runs.values()))
    ratio = first["n_test"] / first["n_train"]
    primary = evaluate(runs, PRIMARY, seeds, ratio, args.exploratory)
    secondary = evaluate(runs, SECONDARY, seeds, ratio, args.exploratory)
    complete = all((arm, seed) in runs for arm in ARMS for seed in seeds)
    if args.exploratory:
        status = "EXPLORATORY — not a confirmatory result"
    elif problems:
        status = "INVALID — consistency problems"
    elif complete:
        status = f"CONFIRMATORY — all {len(ARMS) * len(seeds)} declared fits present"
    else:
        status = (f"INCOMPLETE — {len(runs)} of {len(ARMS) * len(seeds)} declared fits present; "
                  "finish the runs before drawing conclusions")
    result = {
        "plan": PLAN, "status": status, "exploratory": args.exploratory, "problems": problems,
        "seeds": seeds, "alpha": ALPHA, "metric": METRIC, "n_train": first["n_train"], "n_test": first["n_test"],
        "ratio": ratio, "arms": arm_summary(runs, seeds), "primary": primary, "secondary": secondary,
        "ignored": [{"path": path, "reason": reason} for path, reason in ignored],
        "script_sha256": sha256_file(__file__), "results_root": str(root),
        "dataset_fingerprint": first["fingerprint"], "source_sha256": first["source_sha256"],
    }
    if args.output:
        output = Path(args.output).expanduser().resolve()
    else:
        output = (root / "report" if (root / "report").is_dir() else root) / "confirmatory"
    output.mkdir(parents=True, exist_ok=True)
    (output / "confirmatory.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    text = render(result)
    (output / "confirmatory.md").write_text(text, encoding="utf-8")
    with (output / "confirmatory_per_seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["arm", "seed", "test_accuracy", "validation_accuracy", "best_epoch", "split_id"])
        for (arm, seed), run in sorted(runs.items()):
            writer.writerow([arm, seed, run["test"][METRIC], run["validation"][METRIC], run["best_epoch"], run["split_id"]])
    print(text)
    print(f"Written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""BCI Competition IV 2a: explicit channel/time axes and session labels."""
from bisect import bisect_right
from pathlib import Path

import numpy as np

from .base import SignalDataset, bandpass_finite_spans, normalize_samples, source_fingerprint
from .registry import register_dataset


CHANNEL_NAMES = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
                 "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]


@register_dataset("eeg", "eeg", {
    "data_dir": "../ml", "subjects": list(range(1, 10)), "sessions": ["T"],
    "labels_dir": None, "window": 1000, "offset_seconds": 0.5,
    "lowcut": 2.0, "highcut": 30.0, "normalization": "per_sample",
    "artifact_policy": "include", "filter_scope": "run",
}, "Four-class BCI IV 2a motor imagery; 22 EEG channels")
def load_eeg(config):
    import mne
    from scipy.io import loadmat

    root = Path(config["data_dir"]).expanduser().resolve()
    subjects = [int(s) for s in config["subjects"]]
    sessions = [str(s).upper() for s in config["sessions"]]
    if not subjects or len(set(subjects)) != len(subjects) or any(s < 1 or s > 9 for s in subjects):
        raise ValueError("BCI IV 2a subjects must be a nonempty unique subset of 1..9")
    if not sessions or len(set(sessions)) != len(sessions) or not set(sessions) <= {"T", "E"}:
        raise ValueError("BCI IV 2a sessions must be a unique nonempty subset of ['T', 'E']")
    if config["artifact_policy"] not in {"include", "exclude"}:
        raise ValueError("artifact_policy must be include or exclude")
    if config["filter_scope"] not in {"run", "trial", "continuous"}:
        raise ValueError("filter_scope must be run, trial, or continuous")
    window = int(config["window"])
    if window < 1 or config["offset_seconds"] < 0:
        raise ValueError("EEG window must be positive and offset_seconds nonnegative")
    signals, labels, groups, ids, sample_sessions, sample_runs = [], [], [], [], [], []
    sources, skipped = [], {"artifact": 0, "out_of_bounds": 0, "nonfinite": 0}
    fs_values, artifact_count = set(), 0
    for subject in sorted(subjects):
        for session in sorted(sessions):
            basename = f"A{subject:02d}{session}"
            path = root / f"{basename}.gdf"
            if not path.is_file():
                raise FileNotFoundError(f"Required EEG recording not found: {path}; set data.data_dir explicitly")
            sources.append(source_fingerprint(path))
            raw = mne.io.read_raw_gdf(str(path), preload=True, verbose="ERROR")
            fs = float(raw.info["sfreq"])
            fs_values.add(fs)
            if len(raw.ch_names) < 22:
                raise ValueError(f"{path.name} has fewer than 22 EEG channels")
            continuous = raw.get_data()[:22].astype(np.float64)
            events, event_dict = mne.events_from_annotations(raw, verbose="ERROR")
            names = {value: key for key, value in event_dict.items()}
            coded_events = [(int(event[0]) - int(raw.first_samp), names[event[2]]) for event in events]
            run_starts = sorted({position for position, code in coded_events if code == "32766"})
            trial_starts = sorted({position for position, code in coded_events if code == "768"})
            rejected = [position for position, code in coded_events if code == "1023"]
            cues = [(position, code) for position, code in coded_events if code in {"769", "770", "771", "772", "783"}]
            external_labels = None
            if any(code == "783" for _, code in cues):
                labels_root = Path(config["labels_dir"]).expanduser() if config["labels_dir"] else root
                labels_path = labels_root / f"{basename}.mat"
                if not labels_path.is_file():
                    raise FileNotFoundError(
                        f"{path.name} contains unknown evaluation cues (783). Supply official {basename}.mat "
                        "with classlabel via data.labels_dir; never infer test labels from cue codes.")
                official = loadmat(labels_path)
                if "classlabel" not in official:
                    raise ValueError(f"{labels_path} must contain official 'classlabel' labels")
                values = np.asarray(official["classlabel"]).reshape(-1)
                if len(values) != len(cues) or not np.isin(values, [1, 2, 3, 4]).all():
                    raise ValueError(f"{labels_path} must contain one label in 1..4 per cue ({len(cues)} cues)")
                external_labels = values.astype(np.int64) - 1
                sources.append(source_fingerprint(labels_path))
            if config["filter_scope"] == "run":
                continuous, _ = bandpass_finite_spans(continuous, fs, config["lowcut"], config["highcut"], run_starts)
            elif config["filter_scope"] == "continuous":
                from scipy.signal import butter, filtfilt
                b, a = butter(4, [config["lowcut"] / (fs / 2), config["highcut"] / (fs / 2)], btype="band")
                continuous = filtfilt(b, a, continuous.astype(np.float32), axis=-1)
            offset = int(round(float(config["offset_seconds"]) * fs))
            for cue_index, (position, code) in enumerate(cues):
                label = int(external_labels[cue_index]) if external_labels is not None else int(code) - 769
                if label not in range(4):
                    raise ValueError(f"Unresolved class label in {path.name} at sample {position}")
                trial_index = bisect_right(trial_starts, position) - 1
                trial_start = trial_starts[trial_index] if trial_index >= 0 else position
                trial_end = trial_starts[trial_index + 1] if trial_index + 1 < len(trial_starts) else continuous.shape[1]
                artifact = any(trial_start <= marker < trial_end for marker in rejected)
                artifact_count += int(artifact)
                if artifact and config["artifact_policy"] == "exclude":
                    skipped["artifact"] += 1
                    continue
                start, stop = position + offset, position + offset + window
                if start < 0 or stop > continuous.shape[1]:
                    skipped["out_of_bounds"] += 1
                    continue
                epoch = continuous[:, start:stop]
                if config["filter_scope"] == "trial":
                    epoch, _ = bandpass_finite_spans(epoch, fs, config["lowcut"], config["highcut"])
                if not np.isfinite(epoch).all():
                    skipped["nonfinite"] += 1
                    continue
                signals.append(epoch.astype(np.float32))
                labels.append(label)
                groups.append(f"A{subject:02d}")
                ids.append(f"{basename}:cue:{cue_index:03d}:sample:{position}")
                sample_sessions.append(session)
                sample_runs.append(f"{basename}:run:{bisect_right(run_starts, position) - 1}")
    if not signals:
        raise ValueError(f"No usable EEG trials loaded from {root}; skipped={skipped}")
    if len(fs_values) != 1:
        raise ValueError("EEG recordings have different sample rates; explicit resampling is required")
    x = normalize_samples(np.stack(signals), config["normalization"])
    return SignalDataset(x, np.asarray(labels), np.asarray(groups), ids, {
        "dataset": "eeg", "modality": "eeg", "num_classes": 4, "sampling_rate": fs_values.pop(),
        "channel_names": CHANNEL_NAMES, "label_names": ["left_hand", "right_hand", "feet", "tongue"],
        "preprocessing": config, "sources": sources, "skipped": skipped,
        "artifact_trials_seen": artifact_count, "sample_sessions": sample_sessions,
        "sample_runs": sample_runs, "protocol_version": "bci2a-v2",
        "offline_zero_phase_filter": True,
    })

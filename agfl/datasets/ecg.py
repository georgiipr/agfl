"""MIT-BIH binary beat task, with subject identity independent of record ID."""
from pathlib import Path
import numpy as np

from .base import SignalDataset, bandpass_finite_spans, normalize_samples, source_fingerprint
from .registry import register_dataset


# Retain the historical requested recording cohort. Lead exclusions are explicit
# and recorded; these lists are dataset configuration, never model selection.
LEGACY_TRAIN_RECORDS = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
                        "122", "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
LEGACY_VALIDATION_RECORDS = ["100", "103", "104", "105", "111", "113", "117", "121", "200", "202",
                             "210", "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]
LABEL_MAP = {"N": 0, "V": 1, "A": 1, "L": 1, "R": 1, "F": 1}


@register_dataset("ecg", "ecg", {
    "data_dir": "../mit-bih-arrhythmia-database-1.0.0",
    "records": sorted(LEGACY_TRAIN_RECORDS + LEGACY_VALIDATION_RECORDS),
    "window": 256, "lowcut": 0.5, "highcut": 45.0, "normalization": "per_sample",
    "lead": "MLII", "missing_lead": "skip", "label_map": LABEL_MAP,
}, "Historical N-versus-V/A/L/R/F task; not an AAMI five-class benchmark")
def load_ecg(config):
    import wfdb

    root = Path(config["data_dir"]).expanduser().resolve()
    records = [str(record) for record in config["records"]]
    if not records or len(set(records)) != len(records):
        raise ValueError("ECG records must be a nonempty unique list")
    if config["missing_lead"] not in {"skip", "error"}:
        raise ValueError("missing_lead must be skip or error")
    window = int(config["window"])
    if window < 2:
        raise ValueError("ECG window must be at least two samples")
    label_map = config["label_map"]
    class_values = sorted(set(label_map.values()))
    if class_values != list(range(len(class_values))) or len(class_values) < 2:
        raise ValueError("ECG label_map must use contiguous integer classes starting at zero")
    signals, labels, groups, ids, record_ids = [], [], [], [], []
    sources, leads, excluded, discarded, sample_rates = [], {}, [], {}, set()
    left = window // 2
    right = window - left
    skipped = {"out_of_bounds": 0, "nonfinite": 0, "unmapped_annotation": 0}
    for record in sorted(records):
        stem = root / record
        for suffix in (".hea", ".dat", ".atr"):
            path = stem.with_suffix(suffix)
            if not path.is_file():
                raise FileNotFoundError(f"Required ECG source missing: {path}; set data.data_dir explicitly")
            sources.append(source_fingerprint(path))
        header = wfdb.rdheader(str(stem))
        channel_names = list(header.sig_name)
        if config["lead"] == "first":
            channel = 0
        elif config["lead"] in channel_names:
            channel = channel_names.index(config["lead"])
        else:
            reason = {"record": record, "reason": f"lead {config['lead']} unavailable", "available_leads": channel_names}
            if config["missing_lead"] == "error":
                raise ValueError(f"Record {record} lacks lead {config['lead']}; available: {channel_names}")
            excluded.append(reason)
            continue
        signal, info = wfdb.rdsamp(str(stem), channels=[channel])
        fs = float(info["fs"])
        sample_rates.add(fs)
        leads[record] = channel_names[channel]
        clean, valid = bandpass_finite_spans(signal.T, fs, config["lowcut"], config["highcut"])
        ann = wfdb.rdann(str(stem), "atr")
        for position, symbol in zip(ann.sample, ann.symbol):
            if symbol not in label_map:
                skipped["unmapped_annotation"] += 1
                discarded[symbol] = discarded.get(symbol, 0) + 1
                continue
            start, stop = int(position) - left, int(position) + right
            if start < 0 or stop > clean.shape[-1]:
                skipped["out_of_bounds"] += 1
                continue
            if not valid[start:stop].all():
                skipped["nonfinite"] += 1
                continue
            signals.append(clean[:, start:stop].astype(np.float32))
            labels.append(label_map[symbol])
            # MIT-BIH has 48 records from 47 people. 201 and 202 are the same
            # person (the dataset's 202.hea and mitdbdir/intro.htm document this).
            groups.append("mitdb:201_202" if record in {"201", "202"} else f"mitdb:{record}")
            ids.append(f"mitdb:{record}:beat:{int(position)}:{symbol}")
            record_ids.append(record)
    if not signals:
        raise ValueError(f"No usable ECG beats loaded from {root}; excluded={excluded}; skipped={skipped}")
    if len(sample_rates) != 1:
        raise ValueError("ECG recordings have different sample rates; explicit resampling is required")
    return SignalDataset(normalize_samples(np.stack(signals), config["normalization"]),
                         np.asarray(labels), np.asarray(groups), ids, {
        "dataset": "ecg", "modality": "ecg", "num_classes": len(class_values),
        "sampling_rate": sample_rates.pop(), "preprocessing": config, "sources": sources,
        "record_leads": leads, "excluded_records": excluded, "skipped": skipped,
        "discarded_annotation_counts": discarded, "sample_records": record_ids,
        "label_map": label_map, "protocol_version": "mitdb-binary-v2",
        "offline_zero_phase_filter": True,
    })

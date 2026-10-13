"""Offline object-detection evaluation for FireSight YOLO datasets.

The evaluator expects YOLO detection labels (class, normalized center x/y,
normalized width/height). A validation sweep never writes to application
settings and is intentionally separate from the runtime API.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from .engine import EXPECTED_NAMES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_ALIASES = {"val": ("val", "valid", "validation"), "test": ("test",), "train": ("train",)}


@dataclass(frozen=True)
class Box:
    class_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float = 1.0


@dataclass(frozen=True)
class Sample:
    path: Path
    ground_truth: tuple[Box, ...]


def box_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = a + b - intersection
    return intersection / union if union > 0 else 0.0


def match_class_predictions(
    predictions: Iterable[Box], ground_truth: Iterable[Box], iou_threshold: float
) -> tuple[list[tuple[float, bool]], list[Box], list[Box]]:
    """Greedy confidence-ordered, one-to-one matching within an image/class."""
    preds = sorted(predictions, key=lambda row: row.confidence, reverse=True)
    targets = list(ground_truth)
    used: set[int] = set()
    ranked: list[tuple[float, bool]] = []
    false_positives: list[Box] = []
    for pred in preds:
        best_index, best_iou = -1, 0.0
        for index, target in enumerate(targets):
            if index in used:
                continue
            overlap = box_iou(pred.xyxy, target.xyxy)
            if overlap > best_iou:
                best_index, best_iou = index, overlap
        matched = best_index >= 0 and best_iou >= iou_threshold
        if matched:
            used.add(best_index)
        else:
            false_positives.append(pred)
        ranked.append((pred.confidence, matched))
    false_negatives = [target for index, target in enumerate(targets) if index not in used]
    return ranked, false_positives, false_negatives


def average_precision(ranked: list[tuple[float, bool]], ground_truth_count: int) -> float:
    if ground_truth_count <= 0:
        return 0.0
    if not ranked:
        return 0.0
    ranked = sorted(ranked, key=lambda row: row[0], reverse=True)
    tp = np.cumsum([int(is_tp) for _, is_tp in ranked], dtype=float)
    fp = np.cumsum([int(not is_tp) for _, is_tp in ranked], dtype=float)
    recall = tp / ground_truth_count
    precision = tp / np.maximum(tp + fp, 1e-12)
    # COCO-style 101-point interpolated precision envelope.
    return float(np.mean([max(precision[recall >= point], default=0.0) for point in np.linspace(0, 1, 101)]))


def class_metrics(
    per_image: list[tuple[list[Box], list[Box]]],
    class_id: int,
    confidence_threshold: float,
    iou_threshold: float,
) -> tuple[dict[str, Any], list[tuple[float, bool]]]:
    tp = fp = fn = ground_count = 0
    ranked_all: list[tuple[float, bool]] = []
    for predictions, targets in per_image:
        class_predictions = [p for p in predictions if p.class_id == class_id and p.confidence >= confidence_threshold]
        class_targets = [g for g in targets if g.class_id == class_id]
        ground_count += len(class_targets)
        ranked, false_positives, false_negatives = match_class_predictions(class_predictions, class_targets, iou_threshold)
        ranked_all.extend(ranked)
        tp += len(ranked) - len(false_positives)
        fp += len(false_positives)
        fn += len(false_negatives)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"class_id": class_id, "class_name": EXPECTED_NAMES[class_id], "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "ground_truth": ground_count}, ranked_all


def evaluate_at_threshold(
    predictions: list[tuple[list[Box], list[Box]]], confidence_threshold: float, iou_threshold: float = 0.5
) -> dict[str, Any]:
    by_class: dict[str, dict[str, Any]] = {}
    aps_by_iou: list[dict[str, float]] = []
    for class_id, name in EXPECTED_NAMES.items():
        row, _ = class_metrics(predictions, class_id, confidence_threshold, iou_threshold)
        ap_values: dict[str, float] = {}
        for ap_iou in np.arange(0.5, 0.951, 0.05):
            _, ranked = class_metrics(predictions, class_id, 0.001, float(ap_iou))
            gt_count = sum(sum(target.class_id == class_id for target in targets) for _, targets in predictions)
            ap_values[f"{ap_iou:.2f}"] = average_precision(ranked, gt_count)
        row["ap50"] = ap_values["0.50"]
        row["ap50_95"] = float(np.mean(list(ap_values.values())))
        by_class[name] = row
    values = list(by_class.values())
    present = [row for row in values if row["ground_truth"] > 0]
    total_tp = sum(row["tp"] for row in values)
    total_fp = sum(row["fp"] for row in values)
    total_fn = sum(row["fn"] for row in values)
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    return {"confidence_threshold": confidence_threshold, "iou_threshold": iou_threshold,
            "tp": total_tp, "fp": total_fp, "fn": total_fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "map50": float(np.mean([row["ap50"] for row in present])) if present else 0.0,
            "map50_95": float(np.mean([row["ap50_95"] for row in present])) if present else 0.0, "by_class": by_class}


def _resolve_split(data_file: Path, definition: dict[str, Any], split: str) -> tuple[Path, Path, list[str]]:
    """Resolve standard YAML paths, then a matching local split folder without modifying YAML."""
    diagnostics: list[str] = []
    yaml_root = data_file.resolve().parent
    configured_root = Path(str(definition.get("path", yaml_root)))
    if not configured_root.is_absolute():
        configured_root = (yaml_root / configured_root).resolve()
    declared = definition.get(split)
    candidates: list[Path] = []
    if isinstance(declared, str):
        declared_path = Path(declared)
        candidates.extend([(configured_root / declared_path).resolve(), (yaml_root / declared_path).resolve()])
    elif isinstance(declared, list):
        candidates.extend((configured_root / Path(str(item))).resolve() for item in declared)
    for candidate in candidates:
        if candidate.is_dir():
            parts = list(candidate.parts)
            image_part = next((index for index in range(len(parts) - 1, -1, -1) if parts[index].lower() == "images"), None)
            labels = Path(*parts[:image_part], "labels", *parts[image_part + 1:]) if image_part is not None else candidate.parent / "labels"
            if labels.is_dir():
                return candidate, labels, diagnostics
    aliases = SPLIT_ALIASES[split]
    for alias in aliases:
        images = yaml_root / alias / "images"
        labels = yaml_root / alias / "labels"
        if images.is_dir() and labels.is_dir():
            if candidates:
                diagnostics.append(f"Configured {split} image path did not resolve; used validated local {alias}/images and {alias}/labels folders.")
            return images.resolve(), labels.resolve(), diagnostics
    raise FileNotFoundError(f"Could not find paired image and label directories for split '{split}' from {data_file}")


def _read_label_file(path: Path, width: int, height: int) -> tuple[Box, ...]:
    boxes: list[Box] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{path.name}:{line_number}: expected class and four normalized box values")
        try:
            class_id = int(values[0])
            cx, cy, box_w, box_h = map(float, values[1:])
        except ValueError as exc:
            raise ValueError(f"{path.name}:{line_number}: label contains a non-numeric value") from exc
        if class_id not in EXPECTED_NAMES:
            raise ValueError(f"{path.name}:{line_number}: class id {class_id} is outside 0..2")
        if (not all(math.isfinite(value) for value in (cx, cy, box_w, box_h)) or box_w <= 0 or box_h <= 0
                or cx - box_w / 2 < 0 or cy - box_h / 2 < 0 or cx + box_w / 2 > 1 or cy + box_h / 2 > 1):
            raise ValueError(f"{path.name}:{line_number}: box is outside valid normalized YOLO bounds")
        boxes.append(Box(class_id, ((cx - box_w / 2) * width, (cy - box_h / 2) * height,
                                    (cx + box_w / 2) * width, (cy + box_h / 2) * height)))
    return tuple(boxes)


def load_samples(data_file: Path, split: str, limit: int | None = None) -> tuple[list[Sample], list[str], str]:
    definition = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    if not isinstance(definition, dict) or int(definition.get("nc", -1)) != len(EXPECTED_NAMES):
        raise ValueError("Dataset YAML must declare nc: 3 and FireSight's three object classes")
    names = definition.get("names")
    if isinstance(names, dict):
        names = [names.get(i, names.get(str(i))) for i in range(len(EXPECTED_NAMES))]
    if [str(name).lower() for name in (names or [])] != [EXPECTED_NAMES[i] for i in range(len(EXPECTED_NAMES))]:
        raise ValueError(f"Dataset class mapping must be {list(EXPECTED_NAMES.values())}")
    images_dir, labels_dir, diagnostics = _resolve_split(data_file, definition, split)
    image_paths = sorted(path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if limit is not None:
        image_paths = image_paths[:max(0, limit)]
    samples: list[Sample] = []
    failures: list[str] = []
    for image_path in image_paths:
        relative = image_path.relative_to(images_dir)
        label_path = (labels_dir / relative).with_suffix(".txt")
        if not label_path.is_file():
            failures.append(f"{relative.as_posix()}: annotation file is missing; sample skipped")
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            failures.append(f"{relative.as_posix()}: image decoder returned no frame; sample skipped")
            continue
        try:
            targets = _read_label_file(label_path, image.shape[1], image.shape[0])
        except (OSError, ValueError) as exc:
            failures.append(f"{relative.as_posix()}: invalid annotation ({exc}); sample skipped")
            continue
        samples.append(Sample(image_path, targets))
        del image
    return samples, diagnostics + failures, f"{data_file.resolve()}#{split}"


def _predict(model: YOLO, samples: list[Sample], *, imgsz: int, device: str, min_confidence: float) -> list[tuple[list[Box], list[Box]]]:
    all_predictions: list[tuple[list[Box], list[Box]]] = []
    for sample in samples:
        image = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise OSError(f"Image failed to decode during inference: {sample.path.name}")
        result = model.predict(image, conf=min_confidence, imgsz=imgsz, device=device or None, verbose=False)[0]
        predictions = [Box(int(box.cls.item()), tuple(float(v) for v in box.xyxy[0].tolist()), float(box.conf.item())) for box in result.boxes]
        all_predictions.append((predictions, list(sample.ground_truth)))
        del image, result
    return all_predictions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_examples(samples: list[Sample], predictions: list[tuple[list[Box], list[Box]]], threshold: float, output: Path, limit: int = 12) -> list[str]:
    directory = output / "review_examples"
    saved: list[str] = []
    for sample, (preds, targets) in zip(samples, predictions):
        selected = [p for p in preds if p.confidence >= threshold]
        false_positive: list[Box] = []
        false_negative: list[Box] = []
        for class_id in EXPECTED_NAMES:
            _, fp, fn = match_class_predictions([p for p in selected if p.class_id == class_id], [g for g in targets if g.class_id == class_id], 0.5)
            false_positive.extend(fp)
            false_negative.extend(fn)
        if not false_positive and not false_negative:
            continue
        image = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            continue
        for box in targets:
            x1, y1, x2, y2 = map(int, box.xyxy)
            cv2.rectangle(image, (x1, y1), (x2, y2), (65, 205, 127), 2)
            cv2.putText(image, f"GT {EXPECTED_NAMES[box.class_id]}", (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .45, (65, 205, 127), 1)
        for box in selected:
            x1, y1, x2, y2 = map(int, box.xyxy)
            color = (65, 115, 235) if box in false_positive else (174, 174, 65)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{EXPECTED_NAMES[box.class_id]} {box.confidence:.2f}", (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{len(saved)+1:03d}_{sample.path.stem[:48]}.jpg"
        if cv2.imwrite(str(destination), image):
            saved.append(str(destination))
        if len(saved) >= limit:
            break
    return saved


def _write_reports(output: Path, report: dict[str, Any], thresholds: list[dict[str, Any]], min_alarm_recall: float) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path, csv_path = output / f"evaluation_{stamp}.json", output / f"evaluation_{stamp}.csv"
    report["report_json"] = str(json_path)
    report["report_csv"] = str(csv_path)
    report["threshold_sweep"] = thresholds
    eligible = [row for row in thresholds if all(row["by_class"][name]["recall"] >= min_alarm_recall for name in ("fire", "smoke"))]
    ranked = sorted(eligible, key=lambda row: (row["macro_f1"], row["macro_recall"]), reverse=True)
    report["minimum_fire_smoke_recall"] = min_alarm_recall
    report["validation_candidate"] = ranked[0]["confidence_threshold"] if ranked and report["split"] == "val" else None
    if report["validation_candidate"] is not None:
        report["candidate_note"] = "Validation-only candidate among thresholds meeting the minimum fire and smoke recall constraint; ranked by macro F1 with macro recall as tie-breaker. Review false positives and review images before any production change."
    elif report["split"] == "val":
        report["candidate_note"] = "No validation candidate met the minimum fire and smoke recall constraint. Review labels, per-class misses, and the threshold sweep; no production change is suggested."
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    fields = ["split", "confidence_threshold", "class_name", "tp", "fp", "fn", "precision", "recall", "f1", "ap50", "ap50_95"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in thresholds:
            for name, metrics in result["by_class"].items():
                writer.writerow({"split": report["split"], "confidence_threshold": result["confidence_threshold"], "class_name": name, **{key: metrics[key] for key in fields if key in metrics}})
    return json_path, csv_path


def evaluate(
    *, data: Path, model_path: Path, output: Path, split: str = "val", imgsz: int = 640,
    confidence: float = 0.4, thresholds: list[float] | None = None, iou: float = 0.5,
    device: str = "", limit: int | None = None, min_alarm_recall: float = 0.90,
) -> dict[str, Any]:
    if split not in {"val", "test"}:
        raise ValueError("Evaluation supports validation or final test splits; threshold selection is validation-only")
    if split == "test" and thresholds:
        raise ValueError("Threshold sweeps are not allowed on the final test split")
    if not 0 <= confidence <= 1 or not 0 < iou <= 1 or imgsz < 32 or not 0 <= min_alarm_recall <= 1:
        raise ValueError("Confidence, IoU, or image-size parameter is outside its valid range")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model does not exist: {model_path}")
    samples, issues, dataset_id = load_samples(data, split, limit)
    if not samples:
        raise ValueError(f"No valid labeled images found for split '{split}'; see dataset paths and annotation files")
    model = YOLO(str(model_path))
    names = model.names
    actual_names = {int(k): str(v).lower() for k, v in names.items()} if isinstance(names, dict) else {i: str(v).lower() for i, v in enumerate(names)}
    if actual_names != EXPECTED_NAMES:
        raise ValueError(f"Model classes {actual_names} do not match FireSight's required mapping {EXPECTED_NAMES}")
    threshold_values = sorted(set(thresholds or [confidence]))
    if any(not 0 <= value <= 1 for value in threshold_values):
        raise ValueError("All threshold-sweep candidates must be between 0 and 1")
    minimum = min([confidence, *threshold_values])
    predictions = _predict(model, samples, imgsz=imgsz, device=device, min_confidence=minimum)
    metric_thresholds = threshold_values if split == "val" else [confidence]
    reports: list[dict[str, Any]] = []
    for value in metric_thresholds:
        summary = evaluate_at_threshold(predictions, value, iou)
        classes = list(summary["by_class"].values())
        summary["macro_f1"] = statistics.fmean(row["f1"] for row in classes)
        summary["macro_recall"] = statistics.fmean(row["recall"] for row in classes)
        reports.append(summary)
    device_used = str(getattr(getattr(model, "predictor", None), "device", device or "Ultralytics auto"))
    report: dict[str, Any] = {
        "schema_version": 1, "evaluated_at_utc": datetime.now(timezone.utc).isoformat(), "dataset_id": dataset_id,
        "split": split, "dataset_samples": len(samples), "skipped_or_diagnostic_count": len(issues),
        "dataset_issues": issues, "model_path": str(model_path.resolve()), "model_sha256": _sha256(model_path),
        "class_mapping": EXPECTED_NAMES, "software": {"python": sys.version.split()[0], "ultralytics": __import__("ultralytics").__version__,
            "opencv": cv2.__version__, "numpy": np.__version__},
        "device_requested": device or "auto", "device_used": device_used, "image_size": imgsz,
        "iou_threshold_for_counts": iou, "minimum_prediction_confidence": minimum,
        "count_matching": "Per-image, per-class greedy confidence-ordered one-to-one IoU matching; AP uses 101-point interpolated precision at IoU 0.50..0.95.",
        "objective": f"Validation candidates must first meet fire and smoke recall >= {min_alarm_recall:.3f}; eligible candidates are ranked by macro F1 with macro recall tie-breaker.",
    }
    report["review_examples"] = _save_examples(samples, predictions, confidence, output)
    json_path, csv_path = _write_reports(output, report, reports, min_alarm_recall)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate FireSight YOLO model against labeled YOLO detection data")
    parser.add_argument("--data", type=Path, default=Path("dataset/data.yaml"))
    parser.add_argument("--model", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/firesight/evaluations"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--min-alarm-recall", type=float, default=0.90, help="Validation candidate constraint applied separately to fire and smoke recall")
    parser.add_argument("--thresholds", default="", help="Comma-separated validation confidence thresholds, e.g. 0.15,0.25,0.4,0.55")
    parser.add_argument("--device", default="", help="Ultralytics device (auto by default, or cpu/0)")
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N image limit for smoke checks; subset metrics are not a full benchmark")
    args = parser.parse_args(argv)
    try:
        thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()] if args.thresholds else None
        report = evaluate(data=args.data, model_path=args.model, output=args.output, split=args.split, imgsz=args.imgsz,
                          confidence=args.confidence, thresholds=thresholds, iou=args.iou, device=args.device, limit=args.limit,
                          min_alarm_recall=args.min_alarm_recall)
    except Exception as exc:
        parser.error(str(exc))
    print(json.dumps({"split": report["split"], "samples": report["dataset_samples"], "device_used": report["device_used"],
                      "validation_candidate": report["validation_candidate"], "report_json": report["report_json"], "report_csv": report["report_csv"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

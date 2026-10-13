"""Repeatable stage-separated local FireSight model/runtime benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .engine import EXPECTED_NAMES, EventConfirmer
from .config import load_config
from .events import EventStore, utc_now


def summarize_ms(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "min": None, "max": None}
    ordered = sorted(values)
    return {"mean": statistics.fmean(ordered), "median": statistics.median(ordered),
            "p95": float(np.percentile(ordered, 95)), "min": ordered[0], "max": ordered[-1]}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_frames(source: Path, count: int):
    if source.is_file():
        capture = cv2.VideoCapture(str(source))
        try:
            if not capture.isOpened():
                raise OSError(f"Could not open benchmark video: {source.name}")
            for _ in range(count):
                started = time.perf_counter()
                ok, frame = capture.read()
                elapsed_ms = (time.perf_counter() - started) * 1000
                if not ok or frame is None or frame.size == 0:
                    break
                yield frame, elapsed_ms
        finally:
            capture.release()
        return
    if source.is_dir():
        paths = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        if not paths:
            raise ValueError(f"No supported images found under {source}")
        for index in range(count):
            path = paths[index % len(paths)]
            started = time.perf_counter()
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            elapsed_ms = (time.perf_counter() - started) * 1000
            if frame is None or frame.size == 0:
                raise OSError(f"Could not decode benchmark image: {path.name}")
            yield frame, elapsed_ms
        return
    raise FileNotFoundError(f"Benchmark source does not exist: {source}")


def _predictor_device(model: YOLO, requested: str) -> str:
    predictor = getattr(model, "predictor", None)
    device = getattr(predictor, "device", None)
    return str(device) if device is not None else (requested or "Ultralytics auto (not initialized)")


def run_benchmark(
    *, model_path: Path, source: Path, output: Path, imgsz: int = 640,
    warmup: int = 5, frames: int = 100, device: str = "",
) -> dict[str, Any]:
    if not model_path.is_file() or not source.exists():
        raise FileNotFoundError("Model and benchmark source must exist")
    if imgsz < 32 or warmup < 1 or frames < 1 or frames > 100_000:
        raise ValueError("Image size, warm-up, or frame count is outside the supported range")
    app_config = load_config()
    model_load_start = time.perf_counter()
    model = YOLO(str(model_path))
    model_load_seconds = time.perf_counter() - model_load_start
    names = model.names
    actual_names = {int(k): str(v).lower() for k, v in names.items()} if isinstance(names, dict) else {i: str(v).lower() for i, v in enumerate(names)}
    if actual_names != EXPECTED_NAMES:
        raise ValueError(f"Model mapping {actual_names} does not match {EXPECTED_NAMES}")
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    warmup_start = time.perf_counter()
    for _ in range(warmup):
        model.predict(dummy, imgsz=imgsz, device=device or None, verbose=False, conf=app_config.confidence_threshold)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_start
    device_used = _predictor_device(model, device)

    decode_ms: list[float] = []
    inference_ms: list[float] = []
    end_to_end_ms: list[float] = []
    verification_ms: list[float] = []
    measured_frames = 0
    cpu_samples: list[float] = []
    peak_rss_mb: float | None = None
    try:
        import psutil
        process = psutil.Process()
        process.cpu_percent(None)
        rss = lambda: process.memory_info().rss / (1024 * 1024)
        peak_rss_mb = rss()
    except ImportError:
        process = None
        rss = None

    # This isolated temporary EventStore measures JPEG + SQLite write cost and
    # cannot touch the application's live database, logs, or evidence directory.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".benchmark-", dir=output.parent) as temporary:
        temp_root = Path(temporary)
        from dataclasses import replace
        temp_config = replace(app_config, project_root=temp_root, event_db=temp_root / "events.sqlite3",
                              evidence_dir=temp_root / "evidence", detection_log=temp_root / "detections.jsonl",
                              output_video_dir=temp_root / "processed")
        logger = logging.getLogger("firesight")
        handlers_before = set(logger.handlers)
        try:
            store = EventStore(temp_config)
            confirmer = EventConfirmer(temp_config.confidence_threshold, temp_config.confirmation_frames)
            evidence_ms: list[float] = []
            source_iterator = iter(_source_frames(source, frames))
            for frame, decode_time in source_iterator:
                start = time.perf_counter()
                if device_used.startswith("cuda"):
                    torch.cuda.synchronize()
                result = model.predict(frame, imgsz=imgsz, device=device or None, verbose=False, conf=app_config.confidence_threshold)[0]
                if device_used.startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                detections = [(int(box.cls.item()), float(box.conf.item()), tuple(float(v) for v in box.xyxy[0].tolist())) for box in result.boxes]
                class_confidence: dict[str, float] = {}
                for class_id, score, _ in detections:
                    class_name = EXPECTED_NAMES[class_id]
                    class_confidence[class_name] = max(class_confidence.get(class_name, 0.0), score)
                verify_start = time.perf_counter()
                confirmed = confirmer.observe(class_confidence)
                verification_ms.append((time.perf_counter() - verify_start) * 1000)
                evidence_elapsed = None
                if confirmed:
                    persist_start = time.perf_counter()
                    for class_name in confirmed:
                        highest = max(score for class_id, score, _ in detections if EXPECTED_NAMES[class_id] == class_name)
                        store.record_event(timestamp=utc_now(), event_class=class_name, confidence=highest,
                                           source="benchmark-fixture", annotated_frame=result.plot(), camera_name="Benchmark fixture")
                    evidence_elapsed = (time.perf_counter() - persist_start) * 1000
                    evidence_ms.append(evidence_elapsed)
                decode_ms.append(decode_time)
                inference_ms.append(elapsed)
                end_to_end_ms.append(decode_time + elapsed + verification_ms[-1] + (evidence_elapsed or 0.0))
                measured_frames += 1
                if process is not None:
                    cpu_samples.append(process.cpu_percent(None))
                    peak_rss_mb = max(peak_rss_mb or 0.0, rss())
            if not measured_frames:
                raise ValueError("The benchmark source did not yield a decodable frame")
            evidence_count = len(evidence_ms)
        finally:
            for handler in tuple(logger.handlers):
                filename = getattr(handler, "baseFilename", None)
                if handler not in handlers_before and filename and Path(filename).resolve().is_relative_to(temp_root.resolve()):
                    logger.removeHandler(handler)
                    handler.close()

    total_seconds = sum(end_to_end_ms) / 1000
    gpu_memory = None
    if torch.cuda.is_available() and device_used.startswith("cuda"):
        gpu_memory = {"peak_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2),
                      "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024 * 1024), 2),
                      "utilization_percent": None, "utilization_note": "GPU utilization requires NVML; only allocator memory is reported."}
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path.resolve()), "model_sha256": sha256(model_path), "class_mapping": EXPECTED_NAMES,
        "source": str(source.resolve()), "source_kind": "video decode" if source.is_file() else "image file decode",
        "conditions": {"image_size": imgsz, "half_precision": False, "warmup_frames": warmup,
                       "measured_frames": measured_frames, "requested_device": device or "auto", "device_used": device_used,
                       "cuda_available": torch.cuda.is_available(), "confirmation_confidence_threshold": temp_config.confidence_threshold,
                       "inference_confidence_threshold": app_config.confidence_threshold,
                       "confirmation_frames": temp_config.confirmation_frames,
                       "note": "One local process, one model instance, no live camera or browser streaming. Evidence-stage timings are recorded only on synthetic confirmed-event transitions and use a disposable SQLite/evidence store."},
        "warmup_seconds": warmup_seconds,
        "model_load_seconds": model_load_seconds,
        "decode_ms": summarize_ms(decode_ms), "pure_model_inference_ms": summarize_ms(inference_ms),
        "confirmation_logic_ms": summarize_ms(verification_ms), "evidence_persistence_ms_when_event": summarize_ms(evidence_ms),
        "evidence_writes": evidence_count, "end_to_end_frame_ms": summarize_ms(end_to_end_ms),
        "effective_fps": round(measured_frames / total_seconds, 3) if total_seconds > 0 else None,
        "process_cpu_percent_mean_sampled": statistics.fmean(cpu_samples) if cpu_samples else None,
        "process_peak_rss_mb": round(peak_rss_mb, 2) if peak_rss_mb is not None else None,
        "gpu_memory": gpu_memory,
        "software": {"python": __import__("sys").version.split()[0], "torch": torch.__version__,
                     "ultralytics": __import__("ultralytics").__version__, "opencv": cv2.__version__, "numpy": np.__version__},
    }
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = output / f"benchmark_{stamp}.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    report["report_json"] = str(destination)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark FireSight inference and runtime stages on local video or images")
    parser.add_argument("--model", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--source", type=Path, default=Path("dataset/valid/images"), help="Video file or directory of images")
    parser.add_argument("--output", type=Path, default=Path("runs/firesight/benchmarks"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--device", default="", help="Ultralytics device (auto by default, or cpu/0)")
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(model_path=args.model, source=args.source, output=args.output,
                               imgsz=args.imgsz, warmup=args.warmup, frames=args.frames, device=args.device)
    except Exception as exc:
        parser.error(str(exc))
    print(json.dumps({"device_used": report["conditions"]["device_used"], "frames": report["conditions"]["measured_frames"],
                      "inference_ms": report["pure_model_inference_ms"], "effective_fps": report["effective_fps"],
                      "report_json": report["report_json"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

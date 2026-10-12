from __future__ import annotations

import threading
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from ultralytics import YOLO

from .config import AppConfig
from .events import EventStore, sanitize_source, utc_now


EXPECTED_NAMES = {0: "fire", 1: "person", 2: "smoke"}
ALARM_CLASSES = {"fire", "smoke"}


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class FrameResult:
    annotated_frame: np.ndarray
    detections: tuple[Detection, ...]
    events: tuple[dict, ...]
    speed_ms: dict[str, float]
    timestamp: str
    alert_state: str = "NORMAL"


class EventConfirmer:
    """Require an above-threshold detection in N consecutive frames; one event per run."""

    def __init__(self, threshold: float, confirmation_frames: int) -> None:
        self.threshold = threshold
        self.confirmation_frames = confirmation_frames
        self._streaks = {name: 0 for name in ALARM_CLASSES}
        self._active: set[str] = set()

    def reset(self) -> None:
        self._streaks = {name: 0 for name in ALARM_CLASSES}
        self._active.clear()

    @property
    def active(self) -> set[str]:
        return set(self._active)

    @property
    def pending(self) -> dict[str, int]:
        return {name: streak for name, streak in self._streaks.items() if streak and name not in self._active}

    def observe(self, class_confidence: dict[str, float]) -> list[str]:
        confirmed: list[str] = []
        for name in ALARM_CLASSES:
            confidence = class_confidence.get(name, 0.0)
            if confidence >= self.threshold:
                self._streaks[name] += 1
                if self._streaks[name] >= self.confirmation_frames and name not in self._active:
                    self._active.add(name)
                    confirmed.append(name)
            else:
                self._streaks[name] = 0
                self._active.discard(name)
        return sorted(confirmed)


class FireSightEngine:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.events = EventStore(config)
        self.confirmer = EventConfirmer(config.confidence_threshold, config.confirmation_frames)
        self.model: YOLO | None = None
        self._model_lock = threading.Lock()
        if config.external_alerts_enabled:
            logging.getLogger("firesight").warning(
                "External alerts were requested, but no provider is configured; no external messages will be sent."
            )

    def load_model(self) -> dict[int, str]:
        with self._model_lock:
            if self.model is None:
                if not self.config.model_path.is_file():
                    raise FileNotFoundError(f"Model not found: {self.config.model_path}")
                model = YOLO(str(self.config.model_path))
                names = model.names
                normalized = {int(key): str(value).lower() for key, value in names.items()} if isinstance(names, dict) else {i: str(v).lower() for i, v in enumerate(names)}
                if normalized != EXPECTED_NAMES:
                    raise ValueError(f"Unexpected model classes {normalized}; expected {EXPECTED_NAMES}")
                self.model = model
            return dict(EXPECTED_NAMES)

    def reset_session(self) -> None:
        self.confirmer.reset()

    def process_frame(self, frame: np.ndarray, source: str | int, camera_name: str | None = None) -> FrameResult:
        if frame is None or frame.size == 0:
            raise ValueError("Cannot run inference on an empty frame")
        if self.model is None:
            self.load_model()
        assert self.model is not None
        result = self.model.predict(frame, conf=self.config.confidence_threshold, verbose=False)[0]
        annotated = result.plot()
        timestamp = utc_now()
        detections: list[Detection] = []
        max_confidence: dict[str, float] = {}
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = EXPECTED_NAMES.get(class_id, str(class_id))
            confidence = float(box.conf.item())
            xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
            detections.append(Detection(class_id, class_name, confidence, xyxy))
            max_confidence[class_name] = max(max_confidence.get(class_name, 0.0), confidence)

        newly_confirmed = self.confirmer.observe(max_confidence)
        event_rows: list[dict] = []
        event_by_class: dict[str, dict] = {}
        if self.config.internal_alerts_enabled:
            for class_name in newly_confirmed:
                row = self.events.record_event(
                    timestamp=timestamp,
                    event_class=class_name,
                    confidence=max_confidence[class_name],
                    source=source,
                    annotated_frame=annotated,
                    camera_name=camera_name or self.config.camera_name,
                )
                event_rows.append(row)
                event_by_class[class_name] = row

        for detection in detections:
            event_row = event_by_class.get(detection.class_name)
            self.events.log_detection(
                timestamp=timestamp,
                detection_class=detection.class_name,
                confidence=detection.confidence,
                source=source,
                confirmed_event=event_row is not None,
                evidence_path=event_row["evidence_path"] if event_row else None,
                camera_name=camera_name or self.config.camera_name,
            )

        return FrameResult(
            annotated_frame=annotated,
            detections=tuple(detections),
            events=tuple(event_rows),
            speed_ms={key: float(value) for key, value in result.speed.items()},
            timestamp=timestamp,
            alert_state="CRITICAL" if self.confirmer.active else "WARNING" if self.confirmer.pending else "NORMAL",
        )

    def process_image(self, image_path: str | Path) -> tuple[FrameResult, Path]:
        self.reset_session()
        image_path = Path(image_path)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        result = self.process_frame(frame, str(image_path))
        output_dir = self.config.project_root / "runs" / "firesight" / "annotated_images"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_path = output_dir / f"{image_path.stem}_{stamp}.jpg"
        if not cv2.imwrite(str(output_path), result.annotated_frame):
            raise OSError(f"Could not save annotated image: {output_path}")
        return result, output_path


@dataclass(frozen=True)
class VideoSummary:
    frames_processed: int
    output_path: Path | None
    source: str


def _capture_source(source: str | int) -> str | int:
    if isinstance(source, int):
        return source
    text = str(source).strip()
    if text.isdecimal():
        return int(text)
    return text


def _open_capture(capture_source: str | int, is_rtsp: bool) -> cv2.VideoCapture:
    if is_rtsp:
        try:
            return cv2.VideoCapture(
                capture_source,
                cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
            )
        except (TypeError, cv2.error):
            return cv2.VideoCapture(capture_source)

    capture = cv2.VideoCapture(capture_source)
    # Try the platform's native camera backends if the default backend cannot open index 0.
    if capture_source != 0 or capture.isOpened():
        return capture
    capture.release()
    for backend_name in ("CAP_MSMF", "CAP_DSHOW"):
        backend = getattr(cv2, backend_name, None)
        if backend is None:
            continue
        candidate = cv2.VideoCapture(capture_source, backend)
        if candidate.isOpened():
            return candidate
        candidate.release()
        capture = candidate
    return capture


def test_video_connection(source: str | int) -> tuple[int, int]:
    """Probe a stream with one frame and always release it; does not run inference."""
    capture_source = _capture_source(source)
    is_rtsp = "rtsp://" in str(source).lower() or "rtsps://" in str(source).lower()
    capture = _open_capture(capture_source, is_rtsp)
    try:
        if not capture.isOpened():
            raise OSError("Unable to connect to camera. Check the RTSP URL, network, and credentials.")
        ok, frame = capture.read()
        if not ok or frame is None or not frame.size:
            raise OSError("Camera connected but did not provide a readable video frame.")
        height, width = frame.shape[:2]
        return width, height
    finally:
        capture.release()


def run_video_source(
    engine: FireSightEngine,
    source: str | int,
    *,
    stop_event: threading.Event | None = None,
    save_output: bool = False,
    on_frame: Callable[[FrameResult], None] | None = None,
    camera_name: str | None = None,
    on_status: Callable[[str, str | None], None] | None = None,
    reconnect_attempts: int = 2,
) -> VideoSummary:
    """Process a file or RTSP source, optionally writing an annotated MP4."""
    source_for_log = sanitize_source(source)
    engine.reset_session()
    stop_event = stop_event or threading.Event()
    capture_source = _capture_source(source)
    is_rtsp = "rtsp://" in str(source).lower() or "rtsps://" in str(source).lower()
    writer: cv2.VideoWriter | None = None
    output_path: Path | None = None
    frame_count = 0
    capture: cv2.VideoCapture | None = None
    attempts = 0
    final_status = "OFFLINE"
    connected_frame_received = False

    def open_stream() -> cv2.VideoCapture:
        nonlocal attempts
        while not stop_event.is_set():
            try:
                candidate = _open_capture(capture_source, is_rtsp)
            except Exception as exc:
                if attempts >= reconnect_attempts:
                    raise OSError("Unable to connect to camera. Check the RTSP URL, network, and credentials.") from exc
                attempts += 1
                if on_status: on_status(f"RECONNECTING ({attempts}/{reconnect_attempts})", "Camera connection failed; retrying.")
                if stop_event.wait(min(0.5 * attempts, 2.0)): break
                continue
            if candidate.isOpened():
                return candidate
            candidate.release()
            if not is_rtsp or attempts >= reconnect_attempts:
                raise OSError("Unable to connect to camera. Check the RTSP URL, network, and credentials.")
            attempts += 1
            if on_status: on_status(f"RECONNECTING ({attempts}/{reconnect_attempts})", "Camera connection failed; retrying.")
            if stop_event.wait(min(0.5 * attempts, 2.0)): break
        raise OSError("Camera connection cancelled.")

    try:
        if on_status is not None:
            on_status("CONNECTING", None)
        capture = open_stream()
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0 or fps > 240:
            fps = 20.0
        while not stop_event.is_set():
            started = time.monotonic()
            ok, frame = capture.read()
            if not ok:
                connected_frame_received = False
                capture.release()
                capture = None
                is_camera = isinstance(capture_source, int)
                if not is_rtsp or stop_event.is_set():
                    if is_camera and not stop_event.is_set():
                        final_status = "ERROR"
                        raise OSError("Camera stream disconnected or stopped providing frames.")
                    break
                if attempts >= reconnect_attempts:
                    final_status = "ERROR"
                    raise OSError("Camera stream disconnected after the reconnect limit was reached.")
                attempts += 1
                if on_status: on_status(f"RECONNECTING ({attempts}/{reconnect_attempts})", "Camera stream disconnected; retrying.")
                if stop_event.wait(min(0.5 * attempts, 2.0)): break
                capture = open_stream()
                continue
            if not connected_frame_received:
                # An opened VideoCapture is not a healthy source until it yields
                # a real, non-empty frame that the pipeline can process.
                if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                    raise OSError("Camera connected but did not provide a usable video frame.")
                if on_status is not None:
                    on_status("CONNECTED", None)
                connected_frame_received = True
            result = engine.process_frame(frame, source_for_log, camera_name=camera_name)
            if save_output and writer is None:
                engine.config.output_video_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                output_path = engine.config.output_video_dir / f"firesight_{stamp}.mp4"
                height, width = result.annotated_frame.shape[:2]
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    writer.release()
                    writer = None
                    raise OSError(f"Could not create output video: {output_path}")
            if writer is not None:
                writer.write(result.annotated_frame)
            frame_count += 1
            if on_frame is not None:
                on_frame(result)
            delay = max(0.0, 1.0 / fps - (time.monotonic() - started))
            if delay and stop_event.wait(delay):
                break
    except OSError:
        final_status = "ERROR"
        raise
    finally:
        if capture is not None: capture.release()
        if writer is not None:
            writer.release()
        if on_status is not None:
            on_status(final_status, None)
    return VideoSummary(frame_count, output_path, source_for_log)

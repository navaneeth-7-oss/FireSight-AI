"""Local-first HTTP API reusing FireSight's existing inference and event store.

The development server binds to loopback by default. It intentionally does not
pretend to provide authentication; do not expose it to an untrusted network.
"""
from __future__ import annotations

import asyncio
import base64
import csv
import io
import json
import logging
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .config import AppConfig, build_rtsp_url, load_config
from .engine import FireSightEngine, FrameResult, run_video_source, test_video_connection
from .events import sanitize_source
from .secrets import CameraSecretStore, SecretStoreUnavailable

LOG = logging.getLogger("firesight.api")
CameraKind = Literal["webcam", "rtsp", "video"]
IncidentState = Literal["new", "acknowledged", "investigating", "resolved"]


def redact_diagnostic(message: str) -> str:
    def clean_url(match: re.Match[str]) -> str:
        return sanitize_source(match.group(0))

    value = re.sub(r"(?i)rtsps?://[^\s\"'<>]+", clean_url, message)
    return re.sub(r"(?i)(password|passwd|token|secret)=([^&\s]+)", r"\1=[redacted]", value)


class CameraInput(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    source_type: CameraKind
    source: str = Field(min_length=1, max_length=2048)
    username: str = Field(default="", max_length=256)
    password: str = Field(default="", max_length=256)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Camera name cannot be blank")
        return value

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Source cannot be blank")
        return value


class CameraPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    source: str | None = Field(default=None, min_length=1, max_length=2048)
    username: str | None = Field(default=None, max_length=256)
    password: str | None = Field(default=None, max_length=256)

    @field_validator("name")
    @classmethod
    def normalize_optional_name(cls, value: str | None) -> str | None:
        value = value.strip() if value is not None else None
        if value == "":
            raise ValueError("Camera name cannot be blank")
        return value

    @field_validator("source")
    @classmethod
    def normalize_optional_source(cls, value: str | None) -> str | None:
        value = value.strip() if value is not None else None
        if value == "":
            raise ValueError("Source cannot be blank")
        return value


class IncidentPatch(BaseModel):
    status: IncidentState


class SettingsPatch(BaseModel):
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    confirmation_frames: int | None = Field(default=None, ge=1, le=60)
    reconnect_attempts: int | None = Field(default=None, ge=0, le=10)


class CameraTestInput(BaseModel):
    source: str | None = Field(default=None, min_length=1, max_length=2048)
    username: str = Field(default="", max_length=256)
    password: str = Field(default="", max_length=256)


@dataclass
class Camera:
    camera_id: str
    name: str
    source_type: CameraKind
    source: str
    file_label: str | None = None
    status: str = "disconnected"
    last_frame_at: str | None = None
    last_error: str | None = None
    inference_ms: float | None = None
    fps: float | None = None
    frame_number: int = 0
    latest_jpeg: bytes | None = None
    worker: threading.Thread | None = None
    stop_event: threading.Event | None = None
    pause_event: threading.Event | None = None
    last_frame_monotonic: float | None = None
    video_fps: float | None = None
    video_total_frames: int | None = None


class FireSightService:
    """Owns one model instance and bounded camera workers (one active by default)."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.engine = FireSightEngine(config)
        self.lock = threading.RLock()
        self.inference_lock = threading.Lock()
        self.probe_lock = threading.Lock()
        self.image_inference_active = False
        self.model_state = "loading"
        self.model_error: str | None = None
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.settings_path = config.project_root / "runs" / "firesight" / "api_settings.json"
        self.camera_path = config.project_root / "runs" / "firesight" / "api_cameras.json"
        self.secret_store = CameraSecretStore(config.project_root / "runs" / "firesight" / "api_camera_secrets.json")
        self.secret_store_error: str | None = None
        self.settings = self._load_settings()
        self.engine.config = replace(
            config,
            confidence_threshold=self.settings["confidence_threshold"],
            confirmation_frames=self.settings["confirmation_frames"],
        )
        self.engine.events.config = self.engine.config
        self.engine.confirmer.threshold = self.settings["confidence_threshold"]
        self.engine.confirmer.confirmation_frames = self.settings["confirmation_frames"]
        self.cameras = self._load_cameras()
        self.loader = threading.Thread(target=self._load_model, name="firesight-api-model-loader", daemon=True)
        self.loader.start()

    def _load_settings(self) -> dict:
        defaults = {
            "confidence_threshold": self.config.confidence_threshold,
            "confirmation_frames": self.config.confirmation_frames,
            "reconnect_attempts": 2,
            "max_active_sources": 1,
        }
        try:
            raw = json.loads(self.settings_path.read_text(encoding="utf-8"))
            defaults["confidence_threshold"] = float(raw.get("confidence_threshold", defaults["confidence_threshold"]))
            defaults["confirmation_frames"] = int(raw.get("confirmation_frames", defaults["confirmation_frames"]))
            defaults["reconnect_attempts"] = int(raw.get("reconnect_attempts", defaults["reconnect_attempts"]))
            if not 0 <= defaults["confidence_threshold"] <= 1 or not 1 <= defaults["confirmation_frames"] <= 60 or not 0 <= defaults["reconnect_attempts"] <= 10:
                raise ValueError("Invalid saved settings")
        except FileNotFoundError:
            pass
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            LOG.warning("Ignoring invalid API settings (%s)", type(exc).__name__)
        return defaults

    def _load_cameras(self) -> dict[str, Camera]:
        rows = []
        try:
            parsed = json.loads(self.camera_path.read_text(encoding="utf-8"))
            rows = parsed if isinstance(parsed, list) else []
        except FileNotFoundError:
            pass
        except (ValueError, json.JSONDecodeError):
            LOG.warning("Ignoring invalid camera registry file")
        cameras: dict[str, Camera] = {}
        secrets: dict[str, str] = {}
        try:
            secrets = self.secret_store.load()
        except SecretStoreUnavailable:
            self.secret_store_error = "Protected camera secrets could not be recovered for this Windows user. Re-enter the RTSP endpoint and credentials."
        for row in rows:
            try:
                source_type = row["source_type"]
                if source_type not in {"webcam", "rtsp", "video"}:
                    continue
                camera_id = str(row["id"])
                source = str(row["source"])
                if source_type == "rtsp":
                    if camera_id in secrets:
                        source = secrets[camera_id]
                    elif self.secret_store_error:
                        source = sanitize_source(source)
                    elif urlsplit(source).username or urlsplit(source).password or urlsplit(source).query:
                        # Safely migrate legacy full URLs on the next registry write.
                        pass
                camera = Camera(camera_id, str(row["name"]), source_type, source, row.get("file_label"))
                if source_type == "rtsp" and self.secret_store_error:
                    camera.status, camera.last_error = "error", self.secret_store_error
                cameras[camera.camera_id] = camera
            except (KeyError, TypeError):
                continue
        if not cameras:
            if self.config.rtsp_url:
                source, source_type, name = self.config.rtsp_url, "rtsp", self.config.camera_name
            elif self.config.video_source:
                source, source_type, name = self.config.video_source, "video", self.config.camera_name
            else:
                source, source_type, name = "0", "webcam", self.config.camera_name or "Laptop Webcam"
            camera = Camera("default", name, source_type, source)
            cameras[camera.camera_id] = camera
        # Legacy raw URL records are migrated immediately into DPAPI storage.
        if not self.secret_store_error and any(c.source_type == "rtsp" for c in cameras.values()):
            try:
                self._save_cameras(cameras=cameras)
            except HTTPException as exc:
                self.secret_store_error = str(exc.detail)
                for camera in cameras.values():
                    if camera.source_type == "rtsp":
                        camera.source = sanitize_source(camera.source)
                        camera.status, camera.last_error = "error", self.secret_store_error
                self._write_camera_registry(cameras)
        return cameras

    def _save_cameras(self, *, cameras: dict[str, Camera] | None = None) -> None:
        cameras = cameras if cameras is not None else self.cameras
        if self.secret_store_error:
            raise HTTPException(503, self.secret_store_error)
        try:
            self.secret_store.save({c.camera_id: c.source for c in cameras.values() if c.source_type == "rtsp"})
        except SecretStoreUnavailable as exc:
            raise HTTPException(503, "Windows DPAPI secret storage is unavailable; camera credentials were not saved.") from exc
        self._write_camera_registry(cameras)

    def _write_camera_registry(self, cameras: dict[str, Camera]) -> None:
        self.camera_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"id": c.camera_id, "name": c.name, "source_type": c.source_type, "source": sanitize_source(c.source), "file_label": c.file_label}
            for c in cameras.values()
        ]
        temp = self.camera_path.with_suffix(".tmp")
        temp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        temp.replace(self.camera_path)

    def _load_model(self) -> None:
        try:
            self.engine.load_model()
            self.model_state = "ready"
            LOG.info("FireSight model initialized and class mapping validated")
        except Exception as exc:
            self.model_state = "error"
            self.model_error = f"Model initialization failed ({type(exc).__name__})"
            LOG.exception("FireSight model initialization failed")

    def camera_view(self, camera: Camera) -> dict:
        active = bool(camera.worker and camera.worker.is_alive())
        stale = bool(camera.last_frame_monotonic and time.monotonic() - camera.last_frame_monotonic > 5)
        status = "degraded" if active and stale and camera.status == "connected" else camera.status
        return {
            "id": camera.camera_id,
            "name": camera.name,
            "source_type": camera.source_type,
            "status": status,
            "active": active,
            "last_frame_at": camera.last_frame_at,
            "last_error": camera.last_error,
            "inference_ms": camera.inference_ms,
            "fps": camera.fps,
            "frame_number": camera.frame_number,
            "video_total_frames": camera.video_total_frames,
            "video_progress": min(1.0, camera.frame_number / camera.video_total_frames) if camera.video_total_frames else None,
            "video_duration_seconds": round(camera.video_total_frames / camera.video_fps, 2) if camera.video_total_frames and camera.video_fps else None,
            "source_label": "RTSP endpoint configured" if camera.source_type == "rtsp" else camera.file_label or Path(camera.source).name if camera.source_type == "video" else f"Device {camera.source}",
        }

    def get_camera(self, camera_id: str) -> Camera:
        camera = self.cameras.get(camera_id)
        if camera is None:
            raise HTTPException(404, "Camera not found")
        return camera

    def add_camera(self, data: CameraInput) -> Camera:
        source = data.source
        if data.source_type == "rtsp":
            try:
                source = build_rtsp_url(source, data.username, data.password)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        elif data.source_type == "webcam":
            try:
                source = str(int(source))
            except ValueError as exc:
                raise HTTPException(422, "Webcam source must be a device index") from exc
            if int(source) < 0:
                raise HTTPException(422, "Webcam device index cannot be negative")
        elif data.source_type == "video" and not Path(source).expanduser().is_file():
            raise HTTPException(422, "Video source must be an existing local file")
        camera = Camera(uuid.uuid4().hex, data.name, data.source_type, source)
        updated_cameras = {**self.cameras, camera.camera_id: camera}
        self._save_cameras(cameras=updated_cameras)
        self.cameras = updated_cameras
        return camera

    def _frame(self, camera_id: str, result: FrameResult) -> None:
        ok, encoded = cv2.imencode(".jpg", result.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise OSError("Could not encode monitoring frame")
        with self.lock:
            camera = self.get_camera(camera_id)
            now = time.monotonic()
            if camera.last_frame_monotonic is not None:
                delta = now - camera.last_frame_monotonic
                camera.fps = round(1.0 / delta, 2) if delta > 0 else None
            camera.last_frame_monotonic = now
            camera.last_frame_at = result.timestamp
            camera.latest_jpeg = encoded.tobytes()
            camera.inference_ms = round(sum(result.speed_ms.values()), 2) if result.speed_ms else None
            camera.frame_number += 1
            camera.status = "paused" if camera.pause_event is not None and camera.pause_event.is_set() else "connected"
            camera.last_error = None

    def _source_info(self, camera_id: str, fps: float, total_frames: int) -> None:
        with self.lock:
            camera = self.cameras.get(camera_id)
            if camera is not None and camera.source_type == "video":
                camera.video_fps = fps
                camera.video_total_frames = total_frames or None

    def _status(self, camera_id: str, status: str, error: str | None) -> None:
        state = status.strip().lower()
        with self.lock:
            camera = self.cameras.get(camera_id)
            if not camera:
                return
            if state.startswith("reconnecting"):
                state = "reconnecting"
            camera.status = state
            camera.last_error = error
            if state in {"offline", "error"}:
                camera.fps = None

    def connect(self, camera_id: str) -> Camera:
        with self.lock:
            camera = self.get_camera(camera_id)
            if self.model_state != "ready":
                raise HTTPException(503, "Inference model is not ready")
            if self.image_inference_active:
                raise HTTPException(409, "Wait for standalone image inference to finish before connecting a source")
            if camera.worker and camera.worker.is_alive():
                if camera.status in {"connecting", "connected", "reconnecting", "degraded", "paused"}:
                    return camera
            active = [c for c in self.cameras.values() if c.worker and c.worker.is_alive()]
            if any(c.camera_id != camera_id for c in active):
                raise HTTPException(409, "Only one active camera is supported by this inference service")
            camera.stop_event = threading.Event()
            camera.pause_event = threading.Event()
            if camera.source_type == "video":
                camera.frame_number = 0
                camera.video_fps = None
                camera.video_total_frames = None
            camera.status, camera.last_error = "connecting", None
            camera.worker = threading.Thread(
                target=self._run_camera,
                args=(camera.camera_id, camera.stop_event),
                name=f"firesight-camera-{camera.camera_id[:8]}",
                daemon=True,
            )
            camera.worker.start()
            return camera

    def _run_camera(self, camera_id: str, stop_event: threading.Event) -> None:
        with self.lock:
            camera = self.get_camera(camera_id)
            source, name = camera.source, camera.name
        try:
            with self.inference_lock:
                run_video_source(
                    self.engine,
                    source,
                    stop_event=stop_event,
                    pause_event=camera.pause_event,
                    on_frame=lambda result: self._frame(camera_id, result),
                    camera_name=name,
                    on_status=lambda status, error: self._status(camera_id, status, error),
                    on_source_info=lambda fps, frames: self._source_info(camera_id, fps, frames),
                    reconnect_attempts=self.settings["reconnect_attempts"],
                )
        except Exception as exc:
            safe_message = redact_diagnostic(str(exc))
            LOG.error("Camera worker failed camera_id=%s error_type=%s detail=%s", camera_id, type(exc).__name__, safe_message)
            self._status(camera_id, "error", safe_message)
        finally:
            if stop_event.is_set():
                self._status(camera_id, "disconnected", None)

    def disconnect(self, camera_id: str, timeout: float = 6.5) -> Camera:
        with self.lock:
            camera = self.get_camera(camera_id)
            worker, stop_event = camera.worker, camera.stop_event
            if stop_event:
                stop_event.set()
            if not worker or not worker.is_alive():
                camera.status = "disconnected"
                camera.fps = None
                return camera
        if worker is not threading.current_thread():
            worker.join(timeout=timeout)
        with self.lock:
            if worker.is_alive():
                camera.status = "degraded"
                camera.last_error = "Source did not stop within the shutdown timeout"
            else:
                camera.status = "disconnected"
                camera.fps = None
        return camera

    def pause(self, camera_id: str, paused: bool) -> Camera:
        with self.lock:
            camera = self.get_camera(camera_id)
            if camera.source_type != "video":
                raise HTTPException(409, "Pause and resume are supported for local video sources only")
            if not camera.worker or not camera.worker.is_alive() or camera.pause_event is None:
                raise HTTPException(409, "Start the video source before pausing or resuming it")
            if paused:
                camera.pause_event.set()
                camera.status = "paused"
                camera.fps = None
            else:
                camera.pause_event.clear()
                camera.status = "connecting"
            return camera

    def any_active(self) -> bool:
        with self.lock:
            return any(camera.worker and camera.worker.is_alive() for camera in self.cameras.values())

    def shutdown(self) -> None:
        with self.lock:
            active = [camera.camera_id for camera in self.cameras.values() if camera.worker and camera.worker.is_alive()]
        for camera_id in active:
            self.disconnect(camera_id, timeout=6.5)

    def update_settings(self, changes: dict) -> dict:
        new_settings = {**self.settings, **{k: v for k, v in changes.items() if v is not None}}
        if self.any_active() and new_settings != self.settings:
            raise HTTPException(409, "Stop monitoring before changing inference settings")
        payload = {k: new_settings[k] for k in ("confidence_threshold", "confirmation_frames", "reconnect_attempts")}
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.settings_path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp.replace(self.settings_path)
        self.settings = new_settings
        self.engine.config = replace(
            self.engine.config,
            confidence_threshold=new_settings["confidence_threshold"],
            confirmation_frames=new_settings["confirmation_frames"],
        )
        self.engine.events.config = self.engine.config
        self.engine.confirmer.threshold = new_settings["confidence_threshold"]
        self.engine.confirmer.confirmation_frames = new_settings["confirmation_frames"]
        self.engine.reset_session()
        return dict(self.settings)

    @staticmethod
    def validate_source(source_type: CameraKind, source: str) -> str:
        source = source.strip()
        if not source:
            raise HTTPException(422, "Source cannot be blank")
        if source_type == "rtsp":
            try:
                build_rtsp_url(source)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        elif source_type == "webcam":
            try:
                index = int(source)
            except ValueError as exc:
                raise HTTPException(422, "Webcam source must be a device index") from exc
            if index < 0:
                raise HTTPException(422, "Webcam device index cannot be negative")
            source = str(index)
        elif not Path(source).expanduser().is_file():
            raise HTTPException(422, "Video source must be an existing local file")
        return source

    @staticmethod
    def connection_diagnostic(exc: Exception) -> str:
        """Map noisy backend/FFmpeg failures to safe, actionable messages."""
        detail = str(exc).lower()
        if "did not open the stream" in detail or "some cameras do not distinguish" in detail:
            return "The RTSP service is reachable, but the stream was rejected. Verify both the camera credentials and the RTSP path/channel; some camera firmware reports these failures identically."
        if any(term in detail for term in ("401", "403", "unauthorized", "authentication", "auth failed", "forbidden")):
            return "Camera rejected authentication. Verify the RTSP username and password, then test again."
        if any(term in detail for term in ("timed out", "timeout", "deadline")):
            return "Connection timed out. Check that the camera is online, the RTSP port is reachable on your local network, and the camera permits this client."
        if any(term in detail for term in ("refused", "unreachable", "no route", "name or service", "getaddrinfo", "host not found", "could not be resolved")):
            return "Camera host is unreachable. Check its address, power, Wi-Fi/Ethernet connection, and RTSP port."
        if any(term in detail for term in ("404", "not found", "no video", "invalid data", "stream path", "readable video frame")):
            return "The camera opened but no video frame was available. Check the RTSP stream path/channel and that the camera is streaming video."
        return "Connection failed. Verify the RTSP URL, credentials, local network access, and stream path."


def _safe_event(event: dict) -> dict:
    value = dict(event)
    value["source"] = sanitize_source(value.get("source", ""))
    value["status"] = value.get("status") or ("acknowledged" if value.get("acknowledged") else "new")
    value.pop("evidence_path", None)
    return value


def _event_query(service: FireSightService, *, page: int, page_size: int, query: str, event_type: str, severity: str, status: str, camera: str, start: str | None, end: str | None, sort: str = "newest") -> tuple[list[dict], int]:
    start_value = _canonical_date(start, end=False) if start else None
    end_value = _canonical_date(end, end=True) if end else None
    rows, total = service.engine.events.query_events(
        limit=page_size,
        offset=(page - 1) * page_size,
        query=query,
        event_type=event_type,
        severity=severity,
        status=status,
        camera=camera,
        start=start_value,
        end=end_value,
        sort=sort,
    )
    return [_safe_event(row) for row in rows], total


def _parse_date(value: str, *, end: bool) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(422, "Date filters must be ISO-8601 dates or timestamps") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if len(value) == 10 and end:
        parsed += timedelta(days=1) - timedelta(microseconds=1)
    return parsed


def _canonical_date(value: str, *, end: bool) -> str:
    parsed = _parse_date(value, end=end)
    return parsed.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _cors_origins() -> list[str]:
    configured = os.environ.get("FIRESIGHT_CORS_ORIGINS")
    if configured is None:
        return ["http://127.0.0.1:5173", "http://localhost:5173"]

    origins: list[str] = []
    for raw_origin in configured.split(","):
        origin = raw_origin.strip().rstrip("/")
        try:
            parts = urlsplit(origin)
            port = parts.port
        except ValueError as exc:
            raise ValueError("FIRESIGHT_CORS_ORIGINS must contain valid exact HTTP(S) origins.") from exc
        if (
            parts.scheme.lower() not in {"http", "https"}
            or not parts.hostname
            or parts.netloc.endswith(":")
            or parts.username is not None
            or parts.password is not None
            or parts.path
            or "?" in origin
            or "#" in origin
            or origin == "*"
        ):
            raise ValueError("FIRESIGHT_CORS_ORIGINS must contain valid exact HTTP(S) origins, not paths or wildcards.")
        scheme = parts.scheme.lower()
        host = parts.hostname.lower()
        if ":" in host:
            host = f"[{host}]"
        if port is not None and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
            host += f":{port}"
        normalized = f"{scheme}://{host}"
        if normalized not in origins:
            origins.append(normalized)
    if not origins:
        raise ValueError("FIRESIGHT_CORS_ORIGINS must contain at least one exact HTTP(S) origin.")
    return origins


def create_app(config: AppConfig | None = None) -> FastAPI:
    app_config = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = FireSightService(app_config)
        app.state.firesight = service
        yield
        await asyncio.to_thread(service.shutdown)

    application = FastAPI(
        title="FireSight AI API",
        version="1.0.0",
        description="Local-first API over the existing FireSight inference engine and event store.",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "X-File-Name"],
    )

    def service(request: Request) -> FireSightService:
        return request.app.state.firesight

    @application.get("/api/health")
    def health():
        return {"status": "ok", "service": "firesight-api"}

    @application.get("/api/system/status")
    def system_status(request: Request):
        current = service(request)
        return {
            "process": "ok",
            "model": current.model_state,
            "model_error": current.model_error,
            "uptime_seconds": round((datetime.now(timezone.utc) - datetime.fromisoformat(current.started_at)).total_seconds(), 1),
            "active_sources": sum(bool(c.worker and c.worker.is_alive()) for c in current.cameras.values()),
            "max_active_sources": current.settings["max_active_sources"],
            "storage": {"database_available": current.config.event_db.is_file(), "evidence_available": current.config.evidence_dir.is_dir()},
        }

    @application.get("/api/cameras")
    def cameras(request: Request):
        current = service(request)
        with current.lock:
            return [current.camera_view(camera) for camera in current.cameras.values()]

    @application.post("/api/cameras", status_code=201)
    def create_camera(payload: CameraInput, request: Request):
        current = service(request)
        with current.lock:
            if len(current.cameras) >= 32:
                raise HTTPException(409, "Camera registry limit reached")
            camera = current.add_camera(payload)
            return current.camera_view(camera)

    @application.get("/api/cameras/{camera_id}")
    def camera_detail(camera_id: str, request: Request):
        current = service(request)
        with current.lock:
            return current.camera_view(current.get_camera(camera_id))

    @application.patch("/api/cameras/{camera_id}")
    def patch_camera(camera_id: str, payload: CameraPatch, request: Request):
        current = service(request)
        with current.lock:
            camera = current.get_camera(camera_id)
            if camera.worker and camera.worker.is_alive():
                raise HTTPException(409, "Disconnect the camera before editing it")
            if payload.name is not None:
                camera.name = payload.name.strip()
            if payload.source is not None:
                camera.source = current.validate_source(camera.source_type, payload.source)
            if camera.source_type == "rtsp" and (payload.username is not None or payload.password is not None):
                camera.source = build_rtsp_url(camera.source, payload.username or "", payload.password or "")
            if payload.source is not None or payload.username is not None or payload.password is not None:
                if camera.source_type == "video":
                    camera.file_label = Path(camera.source).name
            current._save_cameras()
            return current.camera_view(camera)

    @application.delete("/api/cameras/{camera_id}", status_code=204)
    def delete_camera(camera_id: str, request: Request):
        current = service(request)
        current.disconnect(camera_id)
        with current.lock:
            current.cameras.pop(camera_id, None)
            current._save_cameras()
        return Response(status_code=204)

    @application.post("/api/cameras/{camera_id}/connect")
    def connect_camera(camera_id: str, request: Request):
        current = service(request)
        camera = current.connect(camera_id)
        return current.camera_view(camera)

    @application.post("/api/cameras/{camera_id}/disconnect")
    def disconnect_camera(camera_id: str, request: Request):
        current = service(request)
        camera = current.disconnect(camera_id)
        return current.camera_view(camera)

    @application.post("/api/cameras/{camera_id}/pause")
    def pause_camera(camera_id: str, request: Request):
        return service(request).camera_view(service(request).pause(camera_id, True))

    @application.post("/api/cameras/{camera_id}/resume")
    def resume_camera(camera_id: str, request: Request):
        return service(request).camera_view(service(request).pause(camera_id, False))

    @application.post("/api/cameras/{camera_id}/test")
    async def test_camera(camera_id: str, request: Request, payload: CameraTestInput | None = None):
        current = service(request)
        camera = current.get_camera(camera_id)
        if (camera.worker and camera.worker.is_alive()) or current.any_active():
            raise HTTPException(409, "Stop active monitoring before probing another capture connection")
        if not await asyncio.to_thread(current.probe_lock.acquire, True, 0.1):
            raise HTTPException(409, "A connection test is already in progress")
        try:
            source = camera.source
            if payload is not None:
                try:
                    source = build_rtsp_url(payload.source or camera.source, payload.username, payload.password)
                except ValueError as exc:
                    raise HTTPException(422, str(exc)) from exc
            width, height = await asyncio.to_thread(test_video_connection, source)
        except Exception as exc:
            return {"ok": False, "status": "error", "diagnostic": current.connection_diagnostic(exc)}
        finally:
            current.probe_lock.release()
        return {"ok": True, "status": "valid_frame_received", "width": width, "height": height}

    @application.post("/api/cameras/test")
    async def test_camera_before_save(payload: CameraTestInput, request: Request):
        current = service(request)
        if not payload.source:
            raise HTTPException(422, "Enter a complete RTSP endpoint before testing.")
        try:
            source = build_rtsp_url(payload.source, payload.username, payload.password)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if current.any_active():
            raise HTTPException(409, "Stop active monitoring before probing another capture connection")
        if not await asyncio.to_thread(current.probe_lock.acquire, True, 0.1):
            raise HTTPException(409, "A connection test is already in progress")
        try:
            width, height = await asyncio.to_thread(test_video_connection, source)
        except Exception as exc:
            return {"ok": False, "status": "error", "diagnostic": current.connection_diagnostic(exc)}
        finally:
            current.probe_lock.release()
        return {"ok": True, "status": "valid_frame_received", "width": width, "height": height}

    @application.post("/api/cameras/upload", status_code=201)
    async def upload_video(request: Request, name: str = Query(min_length=1, max_length=80)):
        current = service(request)
        if current.model_state != "ready":
            raise HTTPException(503, "Inference model is not ready")
        safe_name = name.strip()
        if not safe_name:
            raise HTTPException(422, "Camera name cannot be blank")
        raw_name = request.headers.get("X-File-Name", "video.mp4")
        try:
            raw_name = __import__("urllib.parse", fromlist=["unquote"]).unquote(raw_name)
        except Exception:
            pass
        original_name = raw_name.replace("\\", "/").split("/")[-1].replace("\r", "").replace("\n", "")[:180]
        suffix = Path(original_name).suffix.lower()
        if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"}:
            raise HTTPException(415, "Unsupported video extension")
        max_bytes = 200 * 1024 * 1024
        try:
            if int(request.headers.get("content-length", "0")) > max_bytes:
                raise HTTPException(413, "Video exceeds the 200 MiB upload limit")
        except ValueError:
            raise HTTPException(400, "Invalid content length")
        target_dir = current.config.project_root / "runs" / "firesight" / "uploads"
        target_dir.mkdir(parents=True, exist_ok=True)
        file_id = uuid.uuid4().hex
        target_path = target_dir / f"{file_id}{suffix}"
        temp_path = target_dir / f".{file_id}.upload{suffix}"
        byte_count = 0
        try:
            with temp_path.open("wb") as file:
                async for chunk in request.stream():
                    byte_count += len(chunk)
                    if byte_count > max_bytes:
                        raise HTTPException(413, "Video exceeds the 200 MiB upload limit")
                    file.write(chunk)
            if byte_count == 0:
                raise HTTPException(400, "Uploaded video is empty")
            if current.any_active():
                raise HTTPException(409, "Stop active monitoring before validating a video upload")
            try:
                width, height = await asyncio.to_thread(test_video_connection, str(temp_path))
            except Exception as exc:
                raise HTTPException(422, "Uploaded file did not provide a readable video frame") from exc
            temp_path.replace(target_path)
            with current.lock:
                if len(current.cameras) >= 32:
                    raise HTTPException(409, "Camera registry limit reached")
                camera = current.add_camera(CameraInput(name=safe_name, source_type="video", source=str(target_path)))
                camera.file_label = original_name
                current._save_cameras()
            view = current.camera_view(camera)
            view["validated_resolution"] = f"{width}×{height}"
            return view
        except Exception:
            temp_path.unlink(missing_ok=True)
            target_path.unlink(missing_ok=True)
            raise

    @application.post("/api/detect/image")
    async def detect_image(request: Request):
        current = service(request)
        if current.model_state != "ready":
            raise HTTPException(503, "Inference model is not ready")
        with current.lock:
            if current.any_active() or current.image_inference_active:
                raise HTTPException(409, "Disconnect the active source before running standalone image inference")
        content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
        if content_type not in {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}:
            raise HTTPException(415, "Image must be JPEG, PNG, or WebP")
        max_bytes = 12 * 1024 * 1024
        try:
            if int(request.headers.get("content-length", "0")) > max_bytes:
                raise HTTPException(413, "Image exceeds the 12 MiB upload limit")
        except ValueError:
            raise HTTPException(400, "Invalid content length")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > max_bytes:
                raise HTTPException(413, "Image exceeds the 12 MiB upload limit")
        if not body:
            raise HTTPException(400, "Image upload is empty")
        frame = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise HTTPException(422, "Uploaded file is not a readable image")
        with current.lock:
            if current.any_active() or current.image_inference_active:
                raise HTTPException(409, "Disconnect the active source before running standalone image inference")
            current.image_inference_active = True
        try:
            with current.inference_lock:
                result = await asyncio.to_thread(current.engine.process_frame, frame, "image-upload", "Uploaded image")
                ok, encoded = cv2.imencode(".jpg", result.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        except Exception as exc:
            LOG.exception("Image inference failed")
            raise HTTPException(500, "Image inference failed; check local model diagnostics") from exc
        finally:
            with current.lock:
                current.image_inference_active = False
        if not ok:
            raise HTTPException(500, "Could not encode annotated image")
        return {
            "timestamp": result.timestamp,
            "speed_ms": result.speed_ms,
            "alert_state": result.alert_state,
            "detections": [{"class_name": d.class_name, "confidence": d.confidence, "xyxy": d.xyxy} for d in result.detections],
            "confirmed_events": [{k: v for k, v in row.items() if k != "evidence_path"} for row in result.events],
            "annotated_jpeg_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
        }

    @application.get("/api/cameras/{camera_id}/snapshot")
    def snapshot(camera_id: str, request: Request):
        current = service(request)
        camera = current.get_camera(camera_id)
        if not camera.latest_jpeg:
            raise HTTPException(503, f"No frame available; camera state is {current.camera_view(camera)['status']}")
        return Response(camera.latest_jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store, max-age=0", "X-Frame-Timestamp": camera.last_frame_at or ""})

    @application.get("/api/cameras/{camera_id}/stream")
    async def stream(camera_id: str, request: Request):
        current = service(request)
        current.get_camera(camera_id)

        async def frames():
            seen = -1
            while not await request.is_disconnected():
                camera = current.cameras.get(camera_id)
                if camera is None:
                    return
                with current.lock:
                    sequence, jpeg = camera.frame_number, camera.latest_jpeg
                if jpeg and sequence != seen:
                    seen = sequence
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-store\r\n\r\n" + jpeg + b"\r\n"
                await asyncio.sleep(0.12)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"})

    @application.get("/api/incidents")
    def incidents(
        request: Request,
        page: int = Query(1, ge=1),
        page_size: int = Query(25, ge=1, le=100),
        q: str = Query("", max_length=200),
        event_type: str = Query("", max_length=20),
        severity: str = Query("", max_length=20),
        status: str = Query("", max_length=20),
        camera: str = Query("", max_length=80),
        start: str | None = Query(default=None, max_length=40),
        end: str | None = Query(default=None, max_length=40),
        sort: Literal["newest", "oldest"] = "newest",
    ):
        current = service(request)
        items, total = _event_query(current, page=page, page_size=page_size, query=q, event_type=event_type, severity=severity, status=status, camera=camera, start=start, end=end, sort=sort)
        return {"items": items, "page": page, "page_size": page_size, "total": total, "has_more": page * page_size < total}

    @application.get("/api/alerts")
    def alerts(
        request: Request,
        page: int = Query(1, ge=1),
        page_size: int = Query(25, ge=1, le=100),
        q: str = Query("", max_length=200),
        event_type: str = Query("", max_length=20),
        severity: str = Query("", max_length=20),
        status: str = Query("open", max_length=20),
        camera: str = Query("", max_length=80),
        start: str | None = Query(default=None, max_length=40),
        end: str | None = Query(default=None, max_length=40),
        sort: Literal["newest", "oldest"] = "newest",
    ):
        current = service(request)
        items, total = _event_query(current, page=page, page_size=page_size, query=q, event_type=event_type, severity=severity, status=status, camera=camera, start=start, end=end, sort=sort)
        return {"items": items, "page": page, "page_size": page_size, "total": total, "has_more": page * page_size < total}

    @application.get("/api/incidents/{incident_id}")
    def incident_detail(incident_id: str, request: Request):
        current = service(request)
        event = current.engine.events.get_event(incident_id)
        if event is None:
            raise HTTPException(404, "Incident not found")
        return _safe_event(event)

    @application.patch("/api/incidents/{incident_id}")
    def patch_incident(incident_id: str, payload: IncidentPatch, request: Request):
        current = service(request)
        if not current.engine.events.update_status(incident_id, payload.status):
            raise HTTPException(404, "Incident not found")
        event = current.engine.events.get_event(incident_id)
        return _safe_event(event)

    @application.get("/api/evidence/{evidence_id}")
    def evidence(evidence_id: str, request: Request, download: bool = False):
        current = service(request)
        if not evidence_id or len(evidence_id) > 96 or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in evidence_id):
            raise HTTPException(400, "Invalid evidence identifier")
        event = current.engine.events.get_event(evidence_id)
        if event is None:
            raise HTTPException(404, "Evidence record not found")
        evidence_path = Path(event["evidence_path"]).resolve()
        root = current.config.evidence_dir.resolve()
        try:
            evidence_path.relative_to(root)
        except ValueError as exc:
            LOG.error("Evidence path outside configured root event_id=%s", evidence_id)
            raise HTTPException(404, "Evidence is unavailable") from exc
        if not evidence_path.is_file():
            raise HTTPException(404, "Evidence file is missing")
        media_type = "video/mp4" if evidence_path.suffix.lower() in {".mp4", ".webm", ".mov"} else "image/jpeg"
        return FileResponse(evidence_path, media_type=media_type, filename=evidence_path.name if download else None, headers={"Cache-Control": "private, no-store"})

    @application.get("/api/analytics/summary")
    def analytics(request: Request, days: int = Query(7, ge=1, le=366)):
        current = service(request)
        result = current.engine.events.analytics(days)
        result["cameras"] = [current.camera_view(camera) for camera in current.cameras.values()]
        return result

    @application.get("/api/analytics/export.csv")
    def analytics_csv(request: Request):
        current = service(request)
        def rows():
            for values in (["event_id", "timestamp_utc", "camera", "event_type", "confidence_score", "severity", "status"],):
                output = io.StringIO(newline="")
                csv.writer(output).writerow(values)
                yield output.getvalue()
            for event in current.engine.events.iter_export_events():
                values = [event["event_id"], event["timestamp"], event["camera_name"], event["event_class"], event["confidence"], event["severity"], event.get("status", "new")]
                values = [("'" + str(value)) if str(value).startswith(("=", "+", "-", "@", "\t", "\r")) else value for value in values]
                output = io.StringIO(newline="")
                csv.writer(output).writerow(values)
                yield output.getvalue()
        return StreamingResponse(rows(), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": "attachment; filename=FireSight-events.csv", "Cache-Control": "no-store"})

    @application.get("/api/settings")
    def get_settings(request: Request):
        return dict(service(request).settings)

    @application.patch("/api/settings")
    def patch_settings(payload: SettingsPatch, request: Request):
        return service(request).update_settings(payload.model_dump())

    dist = app_config.project_root / "frontend" / "dist"
    if dist.is_dir():
        application.mount("/", StaticFiles(directory=dist, html=True), name="dashboard")
    return application


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("firesight.api:app", host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    main()

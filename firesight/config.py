from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "firesight_config.json"


def _path(value: str | Path, root: Path) -> Path:
    candidate = Path(value).expanduser()
    return (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def _bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    model_path: Path
    confidence_threshold: float
    confirmation_frames: int
    video_source: str
    rtsp_url: str
    evidence_dir: Path
    event_db: Path
    detection_log: Path
    output_video_dir: Path
    internal_alerts_enabled: bool = True
    external_alerts_enabled: bool = False
    camera_name: str = "Laptop Webcam"
    alarm_enabled: bool = True
    email_alerts_enabled: bool = False
    telegram_alerts_enabled: bool = False
    webhook_alerts_enabled: bool = False

    @property
    def configured_source(self) -> str:
        return self.rtsp_url.strip() or self.video_source.strip()


def build_rtsp_url(url: str, username: str = "", password: str = "") -> str:
    """Apply separately entered credentials to an RTSP URL without persisting them."""
    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in {"rtsp", "rtsps"} or not parts.netloc:
        raise ValueError("Enter a complete rtsp:// or rtsps:// camera URL.")
    host = parts.hostname or ""
    if parts.port:
        host += f":{parts.port}"
    if username or password:
        credentials = quote(username, safe="") + (":" + quote(password, safe="") if password else "") + "@"
    else:
        credentials = ""
    return urlunsplit((parts.scheme, credentials + host, parts.path, parts.query, ""))


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load JSON defaults and apply FIRESIGHT_* environment overrides."""
    config_file = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    if not config_file.is_absolute():
        config_file = (PROJECT_ROOT / config_file).resolve()
    raw: dict[str, Any] = {}
    if config_file.exists():
        with config_file.open("r", encoding="utf-8") as stream:
            loaded = json.load(stream)
        if not isinstance(loaded, dict):
            raise ValueError(f"Configuration must be a JSON object: {config_file}")
        raw = loaded

    root = PROJECT_ROOT
    env = os.environ
    model_path = _path(env.get("FIRESIGHT_MODEL_PATH", raw.get("model_path", "models/best.pt")), root)
    confidence = float(env.get("FIRESIGHT_CONFIDENCE_THRESHOLD", raw.get("confidence_threshold", 0.4)))
    frames = int(env.get("FIRESIGHT_CONFIRMATION_FRAMES", raw.get("confirmation_frames", 3)))
    source = env.get("FIRESIGHT_SOURCE", raw.get("video_source", ""))
    rtsp_url = env.get("FIRESIGHT_RTSP_URL", raw.get("rtsp_url", ""))
    evidence_dir = _path(env.get("FIRESIGHT_EVIDENCE_DIR", raw.get("evidence_dir", "runs/firesight/evidence")), root)
    event_db = _path(env.get("FIRESIGHT_EVENT_DB", raw.get("event_db", "runs/firesight/events.sqlite3")), root)
    detection_log = _path(env.get("FIRESIGHT_DETECTION_LOG", raw.get("detection_log", "runs/firesight/detections.jsonl")), root)
    output_video_dir = _path(env.get("FIRESIGHT_OUTPUT_VIDEO_DIR", raw.get("output_video_dir", "runs/firesight/processed_video")), root)
    internal = _bool(env.get("FIRESIGHT_INTERNAL_ALERTS", raw.get("internal_alerts_enabled", True)))
    external = _bool(env.get("FIRESIGHT_EXTERNAL_ALERTS", raw.get("external_alerts_enabled", False)))
    camera_name = str(raw.get("camera_name", "Laptop Webcam"))
    alarm = _bool(raw.get("alarm_enabled", True))
    email_alerts = _bool(raw.get("email_alerts_enabled", False))
    telegram_alerts = _bool(raw.get("telegram_alerts_enabled", False))
    webhook_alerts = _bool(raw.get("webhook_alerts_enabled", False))

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if frames < 1:
        raise ValueError("confirmation_frames must be at least 1")

    return AppConfig(
        project_root=root,
        model_path=model_path,
        confidence_threshold=confidence,
        confirmation_frames=frames,
        video_source=str(source),
        rtsp_url=str(rtsp_url),
        evidence_dir=evidence_dir,
        event_db=event_db,
        detection_log=detection_log,
        output_video_dir=output_video_dir,
        internal_alerts_enabled=internal,
        external_alerts_enabled=external,
        camera_name=camera_name,
        alarm_enabled=alarm,
        email_alerts_enabled=email_alerts,
        telegram_alerts_enabled=telegram_alerts,
        webhook_alerts_enabled=webhook_alerts,
    )


def save_config(config: AppConfig, *, rtsp_url: str | None = None) -> None:
    """Save editable settings, stripping RTSP userinfo so secrets stay in memory only."""
    from .events import sanitize_source

    payload = {
        "model_path": str(config.model_path.relative_to(config.project_root)) if config.model_path.is_relative_to(config.project_root) else str(config.model_path),
        "confidence_threshold": config.confidence_threshold,
        "confirmation_frames": config.confirmation_frames,
        "video_source": config.video_source,
        "rtsp_url": sanitize_source(rtsp_url if rtsp_url is not None else config.rtsp_url),
        "camera_name": config.camera_name,
        "evidence_dir": str(config.evidence_dir.relative_to(config.project_root)) if config.evidence_dir.is_relative_to(config.project_root) else str(config.evidence_dir),
        "event_db": str(config.event_db.relative_to(config.project_root)) if config.event_db.is_relative_to(config.project_root) else str(config.event_db),
        "detection_log": str(config.detection_log.relative_to(config.project_root)) if config.detection_log.is_relative_to(config.project_root) else str(config.detection_log),
        "output_video_dir": str(config.output_video_dir.relative_to(config.project_root)) if config.output_video_dir.is_relative_to(config.project_root) else str(config.output_video_dir),
        "internal_alerts_enabled": config.internal_alerts_enabled,
        "alarm_enabled": config.alarm_enabled,
        "email_alerts_enabled": config.email_alerts_enabled,
        "telegram_alerts_enabled": config.telegram_alerts_enabled,
        "webhook_alerts_enabled": config.webhook_alerts_enabled,
        "external_alerts_enabled": config.external_alerts_enabled,
    }
    DEFAULT_CONFIG_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

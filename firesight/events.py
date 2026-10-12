from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

from .config import AppConfig


@contextmanager
def _connect(path: Path):
    connection = sqlite3.connect(path, timeout=5)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sanitize_source(source: str | int) -> str:
    """Remove URL credentials before source details reach logs or event records."""
    value = str(source)
    parts = urlsplit(value)
    if parts.scheme and parts.netloc:
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    return value


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return (text or "source")[:48]


def configure_logging(config: AppConfig) -> logging.Logger:
    config.detection_log.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("firesight")
    logger.setLevel(logging.INFO)
    if not any(getattr(handler, "baseFilename", None) == str(config.detection_log) for handler in logger.handlers):
        handler = logging.FileHandler(config.detection_log, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


class EventStore:
    """Local JSON-lines detection log, SQLite event history, and evidence frames."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._logger = configure_logging(config)
        config.event_db.parent.mkdir(parents=True, exist_ok=True)
        config.evidence_dir.mkdir(parents=True, exist_ok=True)
        with _connect(config.event_db) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    evidence_path TEXT NOT NULL,
                    camera_name TEXT NOT NULL DEFAULT 'Camera 1',
                    severity TEXT NOT NULL DEFAULT 'CRITICAL',
                    acknowledged INTEGER NOT NULL DEFAULT 0,
                    acknowledged_at TEXT
                )"""
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(events)")}
            for name, definition in (("camera_name", "TEXT NOT NULL DEFAULT 'Camera 1'"), ("severity", "TEXT NOT NULL DEFAULT 'CRITICAL'"), ("acknowledged", "INTEGER NOT NULL DEFAULT 0"), ("acknowledged_at", "TEXT")):
                if name not in columns:
                    connection.execute(f"ALTER TABLE events ADD COLUMN {name} {definition}")

    def log_detection(
        self,
        *,
        timestamp: str,
        detection_class: str,
        confidence: float,
        source: str | int,
        confirmed_event: bool,
        evidence_path: str | None,
        camera_name: str = "Camera 1",
    ) -> None:
        row = {
            "timestamp": timestamp,
            "detection_class": detection_class,
            "confidence": round(float(confidence), 6),
            "source": sanitize_source(source),
            "confirmed_event": bool(confirmed_event),
            "evidence_path": evidence_path,
            "camera_name": camera_name,
        }
        with self._lock:
            self._logger.info(json.dumps(row, ensure_ascii=False))

    def record_event(
        self,
        *,
        timestamp: str,
        event_class: str,
        confidence: float,
        source: str | int,
        annotated_frame: np.ndarray,
        camera_name: str = "Camera 1",
    ) -> dict[str, Any]:
        source_text = sanitize_source(source)
        event_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{_slug(event_class)}-{uuid.uuid4().hex[:8]}"
        event_time = timestamp.replace(":", "").replace("-", "")[:8]
        evidence_dir = self.config.evidence_dir / event_time / _slug(camera_name)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        evidence_path = evidence_dir / f"{event_id}.jpg"
        if not cv2.imwrite(str(evidence_path), annotated_frame):
            raise OSError(f"Could not write evidence image: {evidence_path}")

        row = {
            "event_id": event_id,
            "timestamp": timestamp,
            "event_class": event_class,
            "confidence": round(float(confidence), 6),
            "source": source_text,
            "evidence_path": str(evidence_path.resolve()),
            "camera_name": camera_name,
            "severity": "CRITICAL",
            "acknowledged": 0,
        }
        metadata_path = evidence_path.with_suffix(".json")
        metadata_path.write_text(json.dumps({**row, "source": source_text}, indent=2), encoding="utf-8")
        with self._lock, _connect(self.config.event_db) as connection:
            connection.execute(
                "INSERT INTO events(event_id,timestamp,event_class,confidence,source,evidence_path,camera_name,severity,acknowledged) VALUES(?,?,?,?,?,?,?,?,0)",
                (event_id, timestamp, event_class, row["confidence"], source_text, row["evidence_path"], camera_name, "CRITICAL"),
            )
            self._logger.info(json.dumps({**row, "confirmed_event": True}, ensure_ascii=False))
        return row

    def recent_events(self, limit: int | None = 100) -> list[dict[str, Any]]:
        with _connect(self.config.event_db) as connection:
            connection.row_factory = sqlite3.Row
            query = "SELECT event_id,timestamp,event_class,confidence,source,evidence_path,camera_name,severity,acknowledged,acknowledged_at FROM events ORDER BY timestamp DESC"
            rows = connection.execute(query if limit is None else query + " LIMIT ?", () if limit is None else (max(1, int(limit)),)).fetchall()
        return [dict(row) for row in rows]

    def acknowledge(self, event_id: str) -> None:
        with _connect(self.config.event_db) as connection:
            connection.execute("UPDATE events SET acknowledged=1, acknowledged_at=? WHERE event_id=?", (utc_now(), event_id))

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
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
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
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
                    acknowledged_at TEXT,
                    status TEXT NOT NULL DEFAULT 'new',
                    resolved_at TEXT
                )"""
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(events)")}
            for name, definition in (("camera_name", "TEXT NOT NULL DEFAULT 'Camera 1'"), ("severity", "TEXT NOT NULL DEFAULT 'CRITICAL'"), ("acknowledged", "INTEGER NOT NULL DEFAULT 0"), ("acknowledged_at", "TEXT"), ("status", "TEXT NOT NULL DEFAULT 'new'"), ("resolved_at", "TEXT")):
                if name not in columns:
                    connection.execute(f"ALTER TABLE events ADD COLUMN {name} {definition}")
            connection.execute("UPDATE events SET status='acknowledged' WHERE acknowledged=1 AND status='new'")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_events_class ON events(event_class)")

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
            query = "SELECT event_id,timestamp,event_class,confidence,source,evidence_path,camera_name,severity,acknowledged,acknowledged_at,status,resolved_at FROM events ORDER BY timestamp DESC"
            rows = connection.execute(query if limit is None else query + " LIMIT ?", () if limit is None else (max(1, int(limit)),)).fetchall()
        return [dict(row) for row in rows]

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        with _connect(self.config.event_db) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT event_id,timestamp,event_class,confidence,source,evidence_path,camera_name,severity,acknowledged,acknowledged_at,status,resolved_at FROM events WHERE event_id=?",
                (event_id,),
            ).fetchone()
        return dict(row) if row else None

    def query_events(
        self,
        *,
        limit: int,
        offset: int,
        query: str = "",
        event_type: str = "",
        severity: str = "",
        status: str = "",
        camera: str = "",
        start: str | None = None,
        end: str | None = None,
        sort: str = "newest",
    ) -> tuple[list[dict[str, Any]], int]:
        """Filter and page persisted events in SQLite instead of loading full history."""
        clauses: list[str] = []
        values: list[Any] = []
        if event_type:
            clauses.append("lower(event_class)=lower(?)")
            values.append(event_type)
        if severity:
            clauses.append("lower(severity)=lower(?)")
            values.append(severity)
        if status:
            if status == "new":
                clauses.append("status='new' AND acknowledged=0")
            elif status == "open":
                clauses.append("status!='resolved'")
            else:
                clauses.append("lower(status)=lower(?)")
                values.append(status)
        if camera:
            clauses.append("lower(camera_name) LIKE lower(?)")
            values.append(f"%{camera}%")
        if query:
            clauses.append("(lower(event_id) LIKE lower(?) OR lower(event_class) LIKE lower(?) OR lower(camera_name) LIKE lower(?) OR lower(source) LIKE lower(?))")
            values.extend([f"%{query}%"] * 4)
        if start:
            clauses.append("timestamp >= ?")
            values.append(start)
        if end:
            clauses.append("timestamp <= ?")
            values.append(end)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        columns = "event_id,timestamp,event_class,confidence,source,evidence_path,camera_name,severity,acknowledged,acknowledged_at,status,resolved_at"
        with _connect(self.config.event_db) as connection:
            connection.row_factory = sqlite3.Row
            total = int(connection.execute("SELECT COUNT(*) FROM events" + where, values).fetchone()[0])
            order = "ASC" if sort == "oldest" else "DESC"
            rows = connection.execute(
                f"SELECT {columns} FROM events{where} ORDER BY timestamp {order}, event_id {order} LIMIT ? OFFSET ?",
                [*values, max(1, min(int(limit), 100)), max(0, int(offset))],
            ).fetchall()
        return [dict(row) for row in rows], total

    def analytics(self, days: int = 7) -> dict[str, Any]:
        days = max(1, min(int(days), 366))
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        with _connect(self.config.event_db) as connection:
            totals = connection.execute(
                "SELECT COUNT(*), SUM(CASE WHEN status IN ('acknowledged','investigating','resolved') OR acknowledged=1 THEN 1 ELSE 0 END), SUM(CASE WHEN status='resolved' THEN 1 ELSE 0 END) FROM events"
            ).fetchone()
            by_class = connection.execute("SELECT lower(event_class), COUNT(*) FROM events GROUP BY lower(event_class)").fetchall()
            by_severity = connection.execute("SELECT lower(severity), COUNT(*) FROM events GROUP BY lower(severity)").fetchall()
            trend = connection.execute(
                "SELECT substr(timestamp,1,10), lower(event_class), COUNT(*) FROM events WHERE timestamp>=? GROUP BY substr(timestamp,1,10),lower(event_class) ORDER BY substr(timestamp,1,10)",
                (cutoff,),
            ).fetchall()
        timeseries: dict[str, dict[str, int]] = {}
        for day, name, count in trend:
            timeseries.setdefault(day, {})[name] = int(count)
        return {
            "total_events": int(totals[0] or 0),
            "acknowledged": int(totals[1] or 0),
            "resolved": int(totals[2] or 0),
            "by_class": {str(name): int(count) for name, count in by_class},
            "by_severity": {str(name): int(count) for name, count in by_severity},
            "days": days,
            "timezone": "UTC",
            "timeseries": [{"date": day, "counts": counts} for day, counts in sorted(timeseries.items())],
        }

    def iter_export_events(self, batch_size: int = 500):
        """Yield persisted rows in bounded batches for CSV export."""
        columns = "event_id,timestamp,camera_name,event_class,confidence,severity,status"
        with _connect(self.config.event_db) as connection:
            cursor = connection.execute(f"SELECT {columns} FROM events ORDER BY timestamp DESC,event_id DESC")
            while rows := cursor.fetchmany(max(1, min(int(batch_size), 5000))):
                for row in rows:
                    yield dict(zip(("event_id", "timestamp", "camera_name", "event_class", "confidence", "severity", "status"), row))

    def update_status(self, event_id: str, status: str) -> bool:
        if status not in {"new", "acknowledged", "investigating", "resolved"}:
            raise ValueError("Unsupported incident status")
        acknowledged = status in {"acknowledged", "investigating", "resolved"}
        now = utc_now()
        with _connect(self.config.event_db) as connection:
            cursor = connection.execute(
                "UPDATE events SET status=?, acknowledged=?, acknowledged_at=CASE WHEN ?=1 THEN COALESCE(acknowledged_at, ?) ELSE acknowledged_at END, resolved_at=CASE WHEN ?='resolved' THEN COALESCE(resolved_at, ?) WHEN ?!='resolved' THEN NULL ELSE resolved_at END WHERE event_id=?",
                (status, int(acknowledged), int(acknowledged), now, status, now, status, event_id),
            )
            return cursor.rowcount > 0

    def acknowledge(self, event_id: str) -> None:
        self.update_status(event_id, "acknowledged")

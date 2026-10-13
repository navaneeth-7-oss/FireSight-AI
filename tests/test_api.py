from __future__ import annotations

import logging
import json
import sqlite3
import shutil
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from firesight.api import FireSightService, create_app, redact_diagnostic
from firesight.config import DEFAULT_CONFIG_PATH, load_config
from firesight.engine import FrameResult
from firesight.events import utc_now
from firesight.secrets import SecretStoreUnavailable


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="firesight-api-test-")
        root = Path(self.temp.name)
        self.config = replace(
            load_config(),
            project_root=root,
            model_path=root / "missing-test-model.pt",
            evidence_dir=root / "evidence",
            event_db=root / "events.sqlite3",
            detection_log=root / "detections.jsonl",
            output_video_dir=root / "processed",
            video_source="",
            rtsp_url="",
        )
        built_frontend = Path(__file__).resolve().parents[1] / "frontend" / "dist"
        if built_frontend.is_dir():
            shutil.copytree(built_frontend, root / "frontend" / "dist")
        self.original_user_config = DEFAULT_CONFIG_PATH.read_bytes() if DEFAULT_CONFIG_PATH.is_file() else None
        self.loaded = threading.Event()
        self.loader_patch = patch("firesight.api.FireSightEngine.load_model", side_effect=self.mark_loaded)
        self.loader_patch.start()
        self.client = TestClient(create_app(self.config))
        self.client.__enter__()
        if not self.loaded.wait(3):
            self.fail("Test model loader did not complete")
        self.service = self.client.app.state.firesight

    def mark_loaded(self):
        self.loaded.set()
        return {0: "fire", 1: "person", 2: "smoke"}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.loader_patch.stop()
        logger = logging.getLogger("firesight")
        for handler in tuple(logger.handlers):
            if getattr(handler, "baseFilename", "").startswith(self.temp.name):
                logger.removeHandler(handler)
                handler.close()
        self.temp.cleanup()

    def make_event(self, *, camera_name="Gate"):
        return self.service.engine.events.record_event(
            timestamp=utc_now(),
            event_class="fire",
            confidence=0.86,
            source="rtsp://viewer:private@camera.local:554/live?token=secret",
            annotated_frame=np.zeros((48, 64, 3), dtype=np.uint8),
            camera_name=camera_name,
        )

    def test_health_model_readiness_and_camera_credentials_are_not_returned(self):
        self.assertEqual(self.client.get("/api/health").json()["status"], "ok")
        dashboard = self.client.get("/")
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn("FireSight AI · Operations", dashboard.text)
        status = self.client.get("/api/system/status").json()
        self.assertEqual(status["model"], "ready")
        response = self.client.post("/api/cameras", json={
            "name": "Gate RTSP", "source_type": "rtsp",
            "source": "rtsp://viewer:private@camera.local:554/live?token=secret",
        })
        self.assertEqual(response.status_code, 201)
        body = response.text
        for secret in ("viewer", "private", "secret"):
            self.assertNotIn(secret, body)
        registry = self.service.camera_path.read_text(encoding="utf-8")
        for secret in ("viewer", "private", "secret"):
            self.assertNotIn(secret, registry)
        protected = self.service.secret_store.path.read_text(encoding="utf-8")
        for secret in ("viewer", "private", "token=secret"):
            self.assertNotIn(secret, protected)
        self.assertNotIn("camera.local", self.client.get("/api/cameras").text)

    def test_rtsp_test_before_save_protects_and_recovers_full_url_after_restart(self):
        payload = {
            "source": "rtsps://camera.local:8554/live?profile=main&token=required",
            "username": "view@home",
            "password": "p:a ss",
        }
        exact_url = "rtsps://view%40home:p%3Aa%20ss@camera.local:8554/live?profile=main&token=required"
        with patch("firesight.api.test_video_connection", return_value=(640, 360)) as probe:
            tested = self.client.post("/api/cameras/test", json=payload)
        self.assertEqual(tested.status_code, 200)
        self.assertTrue(tested.json()["ok"])
        probe.assert_called_once_with(exact_url)
        saved = self.client.post("/api/cameras", json={"name": "Front door", "source_type": "rtsp", **payload})
        self.assertEqual(saved.status_code, 201)
        for secret in ("view@home", "p:a ss", "required"):
            self.assertNotIn(secret, saved.text)
        registry = self.service.camera_path.read_text(encoding="utf-8")
        protected = self.service.secret_store.path.read_text(encoding="utf-8")
        for secret in ("view@home", "p:a ss", "required"):
            self.assertNotIn(secret, registry)
            self.assertNotIn(secret, protected)
        with TestClient(create_app(self.config)) as restarted:
            rows = restarted.get("/api/cameras").json()
            row = next(camera for camera in rows if camera["name"] == "Front door")
            self.assertNotIn("source", row)
            recovered = restarted.app.state.firesight.cameras[row["id"]].source
            self.assertEqual(recovered, exact_url)

    def test_legacy_camera_record_is_preserved_and_migrated_to_dpapi(self):
        legacy_url = "rtsp://legacy-user:legacy-pass@camera.local/live?profile=main&token=legacy-token"
        self.service.camera_path.parent.mkdir(parents=True, exist_ok=True)
        self.service.camera_path.write_text(json.dumps([{
            "id": "legacy-camera", "name": "Legacy gate", "source_type": "rtsp", "source": legacy_url,
        }]), encoding="utf-8")
        with TestClient(create_app(self.config)) as restarted:
            service = restarted.app.state.firesight
            self.assertEqual(service.cameras["legacy-camera"].name, "Legacy gate")
            self.assertEqual(service.cameras["legacy-camera"].source, legacy_url)
            registry = service.camera_path.read_text(encoding="utf-8")
            protected = service.secret_store.path.read_text(encoding="utf-8")
            for sensitive in ("legacy-user", "legacy-pass", "legacy-token", "token=legacy-token"):
                self.assertNotIn(sensitive, registry)
                self.assertNotIn(sensitive, protected)

    def test_rtsp_probe_returns_actionable_redacted_failure(self):
        with patch("firesight.api.test_video_connection", side_effect=OSError(
            "401 unauthorized rtsp://viewer:private@camera.local/live?token=secret"
        )):
            response = self.client.post("/api/cameras/test", json={
                "source": "rtsp://camera.local/live?token=secret", "username": "viewer", "password": "private",
            })
        self.assertEqual(response.status_code, 200)
        self.assertIn("authentication", response.json()["diagnostic"].lower())
        for secret in ("viewer", "private", "secret"):
            self.assertNotIn(secret, response.text)

    def test_rtsp_credentials_are_not_saved_when_dpapi_is_unavailable(self):
        with patch.object(self.service.secret_store, "save", side_effect=SecretStoreUnavailable("unavailable")):
            response = self.client.post("/api/cameras", json={
                "name": "Protected camera", "source_type": "rtsp", "source": "rtsp://camera.local/live",
                "username": "viewer", "password": "private",
            })
        self.assertEqual(response.status_code, 503)
        self.assertIn("DPAPI", response.json()["detail"])
        self.assertNotIn("viewer", response.text)
        self.assertFalse(self.service.camera_path.exists())

    def test_incident_filters_pagination_lifecycle_and_evidence_path_confinement(self):
        first = self.make_event()
        self.make_event(camera_name="Warehouse")
        response = self.client.get("/api/incidents", params={"camera": "Gate", "event_type": "fire", "page_size": 1})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total"], 1)
        self.assertFalse(response.json()["has_more"])
        updated = self.client.patch(f"/api/incidents/{first['event_id']}", json={"status": "investigating"})
        self.assertEqual(updated.json()["status"], "investigating")
        image = self.client.get(f"/api/evidence/{first['event_id']}")
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.headers["content-type"], "image/jpeg")
        self.assertEqual(self.client.get("/api/evidence/bad!id").status_code, 400)
        connection = sqlite3.connect(self.config.event_db)
        try:
            connection.execute("UPDATE events SET evidence_path=? WHERE event_id=?", (str(Path(self.temp.name).parent / "outside.jpg"), first["event_id"]))
            connection.commit()
        finally:
            connection.close()
        denied = self.client.get(f"/api/evidence/{first['event_id']}")
        self.assertEqual(denied.status_code, 404)

    def test_settings_validation_persists_without_rewriting_existing_user_config(self):
        invalid = self.client.patch("/api/settings", json={"confidence_threshold": 1.5})
        self.assertEqual(invalid.status_code, 422)
        updated = self.client.patch("/api/settings", json={"confidence_threshold": 0.55, "confirmation_frames": 4})
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["confidence_threshold"], 0.55)
        self.assertEqual(DEFAULT_CONFIG_PATH.read_bytes() if DEFAULT_CONFIG_PATH.is_file() else None, self.original_user_config)
        self.assertTrue((Path(self.temp.name) / "runs" / "firesight" / "api_settings.json").is_file())

    def test_camera_worker_is_singleton_and_second_active_source_is_rejected(self):
        clip = Path(self.temp.name) / "sample.mp4"
        clip.write_bytes(b"mock video path")
        camera = self.client.post("/api/cameras", json={"name": "Validation video", "source_type": "video", "source": str(clip)}).json()
        other = self.client.post("/api/cameras", json={"name": "Second", "source_type": "webcam", "source": "1"}).json()
        started = threading.Event()
        calls = []

        def fake_worker(_engine, _source, *, stop_event, on_frame, on_status, **_kwargs):
            calls.append(1)
            on_status("CONNECTING", None)
            on_frame(FrameResult(np.zeros((8, 8, 3), dtype=np.uint8), (), (), {"inference": 2.0}, utc_now()))
            on_status("CONNECTED", None)
            started.set()
            stop_event.wait(1)
            on_status("OFFLINE", None)

        with patch("firesight.api.run_video_source", side_effect=fake_worker):
            first = self.client.post(f"/api/cameras/{camera['id']}/connect")
            self.assertEqual(first.status_code, 200)
            self.assertTrue(started.wait(2))
            again = self.client.post(f"/api/cameras/{camera['id']}/connect")
            self.assertEqual(again.status_code, 200)
            conflict = self.client.post(f"/api/cameras/{other['id']}/connect")
            self.assertEqual(conflict.status_code, 409)
            settings_conflict = self.client.patch("/api/settings", json={"reconnect_attempts": 3})
            self.assertEqual(settings_conflict.status_code, 409)
            self.assertEqual(len(calls), 1)
            paused = self.client.post(f"/api/cameras/{camera['id']}/pause")
            self.assertEqual(paused.json()["status"], "paused")
            resumed = self.client.post(f"/api/cameras/{camera['id']}/resume")
            self.assertEqual(resumed.json()["status"], "connecting")
            self.assertEqual(self.client.post(f"/api/cameras/{other['id']}/pause").status_code, 409)
            stopped = self.client.post(f"/api/cameras/{camera['id']}/disconnect")
            self.assertEqual(stopped.status_code, 200)
            self.assertEqual(stopped.json()["status"], "disconnected")

    def test_image_inference_returns_actual_result_and_bounds_invalid_input(self):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        ok, payload = cv2.imencode(".jpg", frame)
        self.assertTrue(ok)
        result = FrameResult(frame, (), (), {"inference": 4.5}, utc_now(), "NORMAL")
        with patch.object(self.service.engine, "process_frame", return_value=result):
            response = self.client.post("/api/detect/image", content=payload.tobytes(), headers={"Content-Type": "image/jpeg"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["detections"], [])
            self.assertTrue(response.json()["annotated_jpeg_base64"])
        invalid = self.client.post("/api/detect/image", content=b"not image", headers={"Content-Type": "image/jpeg"})
        self.assertEqual(invalid.status_code, 422)
        self.assertFalse(self.service.image_inference_active)

    def test_csv_exports_persisted_data_without_formula_execution_prefix(self):
        self.make_event(camera_name="=SUM(1,2)")
        result = self.client.get("/api/analytics/export.csv")
        self.assertEqual(result.status_code, 200)
        self.assertIn("timestamp_utc", result.text)
        self.assertIn("'=SUM(1,2)", result.text)


class DiagnosticRedactionTests(unittest.TestCase):
    def test_redacts_embedded_rtsp_credentials_and_query_tokens(self):
        message = "Failed opening rtsp://viewer:private@camera.local/live?token=secret"
        redacted = redact_diagnostic(message)
        for secret in ("viewer", "private", "secret"):
            self.assertNotIn(secret, redacted)
        self.assertIn("camera.local", redacted)

    def test_connection_diagnostics_distinguish_common_failure_types(self):
        cases = [
            (TimeoutError("connect timeout"), "timed out"),
            (OSError("Camera host is unreachable"), "unreachable"),
            (OSError("401 unauthorized"), "authentication"),
            (OSError("The RTSP service did not open the stream"), "path/channel"),
        ]
        for error, expected in cases:
            with self.subTest(error=type(error).__name__):
                self.assertIn(expected, FireSightService.connection_diagnostic(error).lower())


if __name__ == "__main__":
    unittest.main()

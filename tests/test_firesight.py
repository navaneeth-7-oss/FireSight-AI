from __future__ import annotations

import json
import logging
import shutil
import threading
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from firesight.config import build_rtsp_url, load_config, save_config
from firesight.engine import EventConfirmer, FrameResult, _open_capture, run_video_source, test_video_connection
from firesight.events import EventStore, sanitize_source


class EventConfirmerTests(unittest.TestCase):
    def test_weak_and_nonconsecutive_detections_do_not_confirm(self) -> None:
        confirmer = EventConfirmer(threshold=0.6, confirmation_frames=3)
        self.assertEqual(confirmer.observe({"fire": 0.95}), [])
        self.assertEqual(confirmer.observe({"fire": 0.4}), [])
        self.assertEqual(confirmer.observe({"fire": 0.9}), [])
        self.assertEqual(confirmer.observe({}), [])
        self.assertEqual(confirmer.observe({"fire": 0.9}), [])
        self.assertEqual(confirmer.observe({"fire": 0.9}), [])
        self.assertEqual(confirmer.observe({"fire": 0.9}), ["fire"])

    def test_only_one_event_until_detection_clears(self) -> None:
        confirmer = EventConfirmer(threshold=0.4, confirmation_frames=2)
        self.assertEqual(confirmer.observe({"smoke": 0.7}), [])
        self.assertEqual(confirmer.observe({"smoke": 0.8}), ["smoke"])
        self.assertEqual(confirmer.observe({"smoke": 0.8}), [])
        self.assertEqual(confirmer.observe({}), [])
        self.assertEqual(confirmer.observe({"smoke": 0.8}), [])
        self.assertEqual(confirmer.observe({"smoke": 0.8}), ["smoke"])


class SourceSanitizationTests(unittest.TestCase):
    def test_removes_url_credentials_and_query_secrets(self) -> None:
        self.assertEqual(
            sanitize_source("rtsp://viewer:secret@camera.local:554/live?token=private"),
            "rtsp://camera.local:554/live",
        )


class CameraFailureTests(unittest.TestCase):
    def test_unavailable_camera_is_released_and_reported(self) -> None:
        class UnavailableCapture:
            released = False

            def isOpened(self) -> bool:
                return False

            def release(self) -> None:
                self.released = True

        class EngineStub:
            def reset_session(self) -> None:
                pass

        capture = UnavailableCapture()
        with patch("firesight.engine.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(OSError, "Unable to connect to camera"):
                run_video_source(EngineStub(), 0)
        self.assertTrue(capture.released)

    def test_default_webcam_retries_native_camera_backend(self) -> None:
        class Capture:
            def __init__(self, opened: bool) -> None:
                self.opened = opened
                self.released = False

            def isOpened(self) -> bool:
                return self.opened

            def release(self) -> None:
                self.released = True

        default_backend = Capture(False)
        native_backend = Capture(True)
        backend_id = getattr(cv2, "CAP_MSMF", getattr(cv2, "CAP_DSHOW", None))
        if backend_id is None:
            self.skipTest("No native webcam backend constant in this OpenCV build")
        with patch("firesight.engine.cv2.VideoCapture", side_effect=[default_backend, native_backend]) as factory:
            selected = _open_capture(0, False)
        self.assertIs(selected, native_backend)
        self.assertTrue(default_backend.released)
        self.assertEqual(factory.call_args_list[1].args, (0, backend_id))

    def test_opened_camera_is_released_when_stream_ends(self) -> None:
        class OneFrameCapture:
            released = False
            reads = 0

            def isOpened(self) -> bool:
                return True

            def get(self, _property: int) -> float:
                return 30.0

            def read(self):
                self.reads += 1
                if self.reads == 1:
                    return True, np.zeros((32, 32, 3), dtype=np.uint8)
                return False, None

            def release(self) -> None:
                self.released = True

        class EngineStub:
            def reset_session(self) -> None:
                pass

            def process_frame(self, frame, _source, camera_name=None):
                return FrameResult(frame, (), (), {}, "test-time")

        capture = OneFrameCapture()
        statuses = []
        with patch("firesight.engine.cv2.VideoCapture", return_value=capture):
            summary = run_video_source(EngineStub(), "sample.mp4", on_status=lambda state, _error: statuses.append(state))
        self.assertEqual(summary.frames_processed, 1)
        self.assertTrue(capture.released)
        self.assertEqual(statuses, ["CONNECTING", "CONNECTED", "OFFLINE"])

    def test_opened_camera_is_not_connected_until_it_returns_a_valid_frame(self):
        class OpenWithoutFrame:
            released = False
            def isOpened(self): return True
            def get(self, _property): return 20.0
            def read(self): return True, None
            def release(self): self.released = True
        class EngineStub:
            def reset_session(self): pass
        capture = OpenWithoutFrame()
        statuses = []
        with patch("firesight.engine.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(OSError, "usable video frame"):
                run_video_source(EngineStub(), "sample.mp4", on_status=lambda state, _error: statuses.append(state))
        self.assertNotIn("CONNECTED", statuses)
        self.assertTrue(capture.released)

    def test_rtsp_probe_failure_releases_capture(self):
        class Unavailable:
            released = False
            def isOpened(self): return False
            def release(self): self.released = True
        capture = Unavailable()
        with patch("firesight.engine.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(OSError, "Unable to connect"):
                test_video_connection("rtsp://camera.local/live")
        self.assertTrue(capture.released)

    def test_rtsp_probe_timeout_releases_capture(self):
        class TimedOut:
            released = False
            def isOpened(self): return True
            def read(self): raise cv2.error("read timeout")
            def release(self): self.released = True
        capture = TimedOut()
        with patch("firesight.engine._open_capture", return_value=capture):
            with self.assertRaisesRegex(cv2.error, "timeout"):
                test_video_connection("rtsp://camera.local/live")
        self.assertTrue(capture.released)

    def test_rtsp_stream_disconnect_uses_bounded_reconnect(self):
        class Capture:
            def __init__(self): self.released = False; self.reads = 0
            def isOpened(self): return True
            def get(self, _property): return 20.0
            def read(self):
                self.reads += 1
                return (True, np.zeros((8, 8, 3), dtype=np.uint8)) if self.reads == 1 else (False, None)
            def release(self): self.released = True
        captures = [Capture(), Capture()]
        stop = threading.Event()
        statuses = []
        class EngineStub:
            def reset_session(self): pass
            def process_frame(self, frame, _source, camera_name=None):
                return FrameResult(frame, (), (), {}, "test-time")
        def status(state, _error):
            statuses.append(state)
            if state == "CONNECTED" and statuses.count("CONNECTED") == 2:
                stop.set()
        with patch("firesight.engine._open_capture", side_effect=captures):
            run_video_source(EngineStub(), "rtsp://camera.local/live", stop_event=stop, on_status=status)
        self.assertIn("RECONNECTING (1/2)", statuses)
        self.assertEqual(statuses.count("CONNECTED"), 2)
        self.assertTrue(all(capture.released for capture in captures))


class EventStoreTests(unittest.TestCase):
    def test_event_writes_evidence_database_and_detection_log(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        root = Path(tempfile.mkdtemp(prefix=".firesight-test-", dir=project_root)).resolve()
        try:
            config = replace(
                load_config(),
                evidence_dir=root / "evidence",
                event_db=root / "events.sqlite3",
                detection_log=root / "detections.jsonl",
            )
            store = EventStore(config)
            frame = np.zeros((80, 120, 3), dtype=np.uint8)
            event = store.record_event(
                timestamp="2026-10-08T00:00:00.000Z",
                event_class="fire",
                confidence=0.81,
                source="rtsp://user:password@camera/live",
                annotated_frame=frame,
            )
            self.assertTrue(Path(event["evidence_path"]).is_file())
            self.assertIsNotNone(cv2.imread(event["evidence_path"]))
            self.assertEqual(store.recent_events()[0]["event_class"], "fire")
            store.log_detection(
                timestamp="2026-10-08T00:00:00.000Z",
                detection_class="fire",
                confidence=0.81,
                source="rtsp://user:password@camera/live?token=hidden",
                confirmed_event=True,
                evidence_path=event["evidence_path"],
            )
            rows = [json.loads(line) for line in config.detection_log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[-1]["detection_class"], "fire")
            self.assertTrue(rows[-1]["confirmed_event"])
            self.assertNotIn("password", rows[-1]["source"])
            self.assertNotIn("hidden", rows[-1]["source"])
            store.acknowledge(event["event_id"])
            self.assertEqual(store.recent_events()[0]["acknowledged"], 1)
            self.assertEqual(store.recent_events()[0]["status"], "acknowledged")
            page, total = store.query_events(limit=10, offset=0, status="acknowledged", event_type="fire")
            self.assertEqual(total, 1)
            self.assertEqual(page[0]["event_id"], event["event_id"])
            self.assertTrue(store.update_status(event["event_id"], "resolved"))
            self.assertEqual(store.get_event(event["event_id"])["status"], "resolved")
            self.assertIsNotNone(store.get_event(event["event_id"])["resolved_at"])
            self.assertEqual(store.recent_events()[0]["camera_name"], "Camera 1")
            self.assertEqual(Path(event["evidence_path"]).parent.name, "Camera_1")
        finally:
            logger = logging.getLogger("firesight")
            for handler in tuple(logger.handlers):
                if Path(handler.baseFilename).resolve() == (root / "detections.jsonl").resolve():
                    logger.removeHandler(handler)
                    handler.close()
            if root.name.startswith(".firesight-test-") and project_root in root.parents:
                shutil.rmtree(root)


class ConfigurationTests(unittest.TestCase):
    def test_environment_overrides_source_and_confirmation(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "FIRESIGHT_RTSP_URL": "rtsp://camera.local/live",
                "FIRESIGHT_CONFIDENCE_THRESHOLD": "0.55",
                "FIRESIGHT_CONFIRMATION_FRAMES": "5",
            },
            clear=False,
        ):
            config = load_config()
        self.assertEqual(config.configured_source, "rtsp://camera.local/live")
        self.assertEqual(config.confidence_threshold, 0.55)
        self.assertEqual(config.confirmation_frames, 5)

    def test_rtsp_credentials_are_encoded_and_configured_url_is_redacted(self):
        url = build_rtsp_url("rtsp://camera.local:554/live", "viewer@site", "p:a ss")
        self.assertEqual(url, "rtsp://viewer%40site:p%3Aa%20ss@camera.local:554/live")
        self.assertNotIn("p%3Aa", sanitize_source(url))
        self.assertNotIn("viewer", sanitize_source(url))
        with self.assertRaises(ValueError): build_rtsp_url("")
        with self.assertRaises(ValueError): build_rtsp_url("http://camera.local/stream")

    def test_config_saving_never_persists_credentials(self):
        project_root = Path(__file__).resolve().parents[1]
        root = Path(tempfile.mkdtemp(prefix=".firesight-config-test-", dir=project_root)).resolve()
        config_path = project_root / "firesight_config.json"
        original = config_path.read_bytes()
        try:
            config = replace(load_config(), project_root=project_root, rtsp_url="rtsp://camera.local/live", camera_name="Gate")
            # Exercise the payload builder against a temporary JSON target without touching the checked-in app config.
            with patch("firesight.config.DEFAULT_CONFIG_PATH", root / "config.json"):
                save_config(config, rtsp_url="rtsp://viewer:topsecret@camera.local/live")
            saved = (root / "config.json").read_text(encoding="utf-8")
            self.assertIn("rtsp://camera.local/live", saved)
            self.assertNotIn("topsecret", saved)
            self.assertNotIn("viewer", saved)
        finally:
            self.assertEqual(config_path.read_bytes(), original)
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()

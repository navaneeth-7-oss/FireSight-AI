# FireSight-AI Application Report

Date: 2026-10-08

## Architecture and dashboard

- `firesight_app.py` launches the desktop application.
- `firesight/gui.py` now provides a dark CCTV operations dashboard with Monitor, Event history, and Settings navigation; a large aspect-ratio-preserving annotated live view; camera/model/FPS/resolution/frame-count/last-frame health cards; fire/smoke/person counters; source name/type/status; a recent-events table with acknowledgement state; and an internal alert banner. Source options are visibly available as **WEBCAM**, **CCTV / RTSP**, **VIDEO**, and **IMAGE**. Laptop Webcam is the selected default at startup; CCTV never connects without an explicit **CONNECT CCTV** action.
- Webcam, video, RTSP capture, RTSP test probe, and model inference run on worker threads. Tkinter receives frames and status through queues and renders on its main event loop. Switching sources stops the existing worker and starts the requested source after capture release. Stop/disconnect and window close wait for capture cleanup.
- RTSP status reports OFFLINE, CONNECTING, CONNECTED, RECONNECTING (bounded retry number), and ERROR. Stream-loss recovery is bounded to two retries; **RECONNECT** allows a user-initiated retry. The separate **TEST CONNECTION** probe opens the stream, reads one frame, reports dimensions, and releases it without replacing the active source. OpenCV cannot reliably distinguish authentication from reachability errors, so the UI recommends checking URL, network, and credentials without exposing secrets.
- `firesight/engine.py` loads and validates the trained YOLO classes (`fire`, `person`, `smoke`), annotates detections, and runs image/video/camera inference. Fire and smoke require confidence `>= 0.40` for three consecutive frames by default; only confirmed events enter the critical event workflow. Pending streaks display WARNING, confirmed active detections display CRITICAL, and a cleared condition returns to NORMAL.
- `firesight/events.py` keeps local SQLite event history and JSONL detection logs. Confirmed annotated evidence uses `evidence/YYYYMMDD/<camera>/<event>.jpg` and a JSON metadata sidecar. Records include timestamp, camera, event class, confidence, severity, source, and evidence path. Acknowledgement is stored in SQLite. URL userinfo and query strings are removed from saved source values.
- `firesight/config.py` contains editable settings and builds RTSP URLs from the endpoint and separate username/password fields. Credentials are used in memory and are excluded from saved config, logs, event history, and evidence metadata. `firesight_config.json` stores non-secret defaults.
- The local audible alarm is controlled by settings; **SILENCE ALARM** silences repeat tones for the current alert. **ACKNOWLEDGE ALERT** records acknowledgement without deleting the event. Email, Telegram, and webhook selections are future configuration only. No external sender exists and no external messages are sent.

## Files created or changed for this dashboard upgrade

- Changed: `firesight/gui.py`, `firesight/engine.py`, `firesight/events.py`, `firesight/config.py`, `firesight_config.json`, `tests/test_firesight.py`, and this report.
- Created: `FIRESIGHT_USER_GUIDE.md`.
- Preserved: image, video, webcam, RTSP, existing model checkpoint, and event-confirmation pipeline. Dataset and trained weights were not modified. `models/best.pt` SHA-256 is `33F6641E9DBA8FD0A419ACA1EDED81EAE7D049F4EE794586BEF435600243C10E` (5,459,610 bytes).

## How to run

From PowerShell in the project directory:

```powershell
.\venv\Scripts\Activate.ps1
python firesight_app.py
```

See [FIRESIGHT_USER_GUIDE.md](FIRESIGHT_USER_GUIDE.md) for the exact RTSP connection steps. The interface accepts a local MP4/video or image, opens the default webcam at index 0, and accepts RTSP endpoint and credentials through the CCTV panel. External notification delivery is not implemented.

## Configuration

`firesight_config.json` supports `model_path`, `confidence_threshold`, `confirmation_frames`, `video_source`, `rtsp_url`, `camera_name`, `evidence_dir`, event database/log/output paths, `internal_alerts_enabled`, `alarm_enabled`, `external_alerts_enabled`, and future channel selections `email_alerts_enabled`, `telegram_alerts_enabled`, and `webhook_alerts_enabled`. Defaults are 0.40 confidence, three frames, local alarm on, external alerts off. The Settings page edits the runtime values and saves non-secret settings. Credentials entered into the CCTV panel are not written to JSON.

## Tests performed

- **Automated unit tests:** PASS, 13 tests, including threshold/consecutive-frame confirmation, event de-duplication until a detection clears, RTSP URL validation and credential encoding/redaction, config credential redaction, evidence/database/log writes, acknowledgement persistence, webcam open failure/release, RTSP probe failure and timeout cleanup, bounded mocked RTSP reconnection, and environment overrides.
- **Model load:** PASS in the current venv; `models/best.pt` reports `{0: fire, 1: person, 2: smoke}`.
- **Image inference:** PASS on an existing `dataset/test/images` image; two detections and an annotated result were produced.
- **Video inference:** PASS on a generated one-frame MP4 built from that existing test image; one frame was processed and an annotated MP4 was produced. This is a pipeline check, not independent real-scene validation.
- **Evidence and internal event storage:** PASS through the unit test; the test created and read an annotated image, queried its SQLite event, checked the JSONL log, verified camera metadata, and acknowledged the event.
- **CCTV / RTSP:** IMPLEMENTED. URL validation/composition, credential encoding/redaction, failed-probe cleanup, timeout cleanup, and bounded mocked reconnection tests PASS. **Real CCTV testing: NOT YET AVAILABLE**; no physical CCTV camera or RTSP endpoint was connected.
- **Laptop webcam:** Project-owner supplied status: **PASS — manually verified and currently used for testing.** Our direct OpenCV index-0 verification in this runtime returned `isOpened() = False`, so this session did not process webcam frames and cannot independently reproduce that pass. Webcam support remains available; mocked open/release and error paths pass.
- **GUI startup and visual verification:** PASS. Started the final dashboard, confirmed model status reaches LOADED and system status READY, and inspected the live monitor. The visible dashboard includes source controls, a large feed area, CCTV connect/test/disconnect controls, health cards, counters, and event history. Manually clicked CONNECT CCTV with an empty endpoint: the dashboard displayed a clear URL error and ERROR state without crashing; DISCONNECT returned it to OFFLINE and clears the panel message. (The silence control was exercised in the prior UI iteration; acknowledge persistence is covered by unit test.)
- **Camera disconnect handling:** PASS for mocked camera open failure, timeout cleanup, stream loss, bounded reconnection, and opened-capture release paths. Real CCTV connectivity was unavailable.
- **External alerts:** not sent. No provider adapter is implemented.

## Limitations

- This session could not open webcam index 0 and had no RTSP CCTV endpoint. Real CCTV connectivity, authentication, sustained stream stability, production FPS, and camera-specific RTSP behavior remain unverified. Webcam PASS above is project-owner reported, not reproduced in this session.
- The local alarm uses Tkinter's system bell, which may be quiet or disabled by the operating system.
- Email, Telegram, and webhook delivery are configuration placeholders only. The dashboard deliberately sends no external notifications.
- Video smoke testing used a generated one-frame clip from a test image. It does not validate live scene accuracy or long-duration recording.
- Inference performance depends on the installed PyTorch device/runtime. The configured test environment uses CPU-only Torch.

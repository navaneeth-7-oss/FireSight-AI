# FireSight AI User Guide

## Launch

From PowerShell in the project directory:

```powershell
.\venv\Scripts\Activate.ps1
python firesight_app.py
```

FireSight is a local Tkinter desktop application. The YOLO model loads in a background thread. No camera starts automatically.

## Live Monitor

The Live Monitor has three status cards: Active Camera, Live Performance (measured FPS and resolution), and Current Detections (fire, smoke, and person counts for the latest processed frame). Confirmed fire/smoke incidents appear in the alert banner and create event records; current detection boxes are not confirmed incidents.

Choose Webcam, CCTV / RTSP, Video, or Image from the source controls. Video and image keep their native file picker. STOP SOURCE is available while a source is processing. Switching a source stops and releases the existing capture before starting another. Enable Save processed video before starting a video or camera source if an annotated output is wanted.

## Camera Sources

Choose Camera Sources in the sidebar to configure CCTV. Enter a camera name, RTSP endpoint, and optional separate username/password. Credentials are masked and are used in memory; they are not saved to the configuration. Use TEST CONNECTION to probe independently, CONNECT CAMERA to start inference, DISCONNECT to stop, and RECONNECT after an error. Errors do not reveal the credential-bearing URL. RTSP retries are bounded. No physical CCTV endpoint was available for an end-to-end test in this environment.

## Alerts and Events

Alerts lists confirmed incident records. Acknowledge records that an operator reviewed an alert. Silence Local Alarm only mutes the audible notification; neither action resolves or deletes an incident.

Events contains the full stored history with text search, camera, event-type, and acknowledgement filters, and 100-row pagination. Double-click a row or select it and choose View Selected Evidence. The in-app viewer fits image evidence without distortion, shows event metadata, and has a View Original action. Missing or unsupported files produce an error message. Evidence references are restricted to the configured evidence folder or the project directory; existing files are not moved.

## Analytics, Settings, and Health

Historical overview counts confirmed event records by class and lists cameras found in the event history. These values come from the locally stored event database.

Settings retains the existing confidence threshold, consecutive-frame confirmation, source defaults, alarm, notifications, and evidence settings. The model weights path is editable and applies on the next launch. Credentials are never saved. External email, Telegram, and webhook delivery are configuration placeholders; no sender is implemented.

System Health displays model load state, camera state, frame processing, and the evidence directory. Detailed diagnostics and external notification delivery are not implemented.

## Data and limitations

Events are stored in SQLite; detection logs and annotated evidence remain under the configured local `runs/firesight/` paths by default. The interface redesign did not change model weights, event records, evidence files, or the detection engine. Webcam/CCTV operation still depends on local device permissions and reachable camera hardware. The app has no user authentication or remote web interface and is not a public-deployment build.

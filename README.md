# FireSight AI

FireSight AI is a local fire and smoke monitoring application. It uses the existing YOLO model, OpenCV capture pipeline, consecutive-frame event confirmer, SQLite event store, and evidence images. The repository now includes both the existing Tkinter desktop app and a local-first React/TypeScript dashboard backed by a FastAPI service.

The web API binds to `127.0.0.1` by default and has no authentication. It is intended for a trusted local workstation and engineering demonstration. Do not expose it to a network until authentication, TLS, access controls, and an operational security review are in place.

## Architecture

- `firesight/engine.py` owns the existing YOLO inference and capture processing.
- `firesight/events.py` owns SQLite event persistence, detection logs, evidence images, lifecycle status, and paged queries.
- `firesight/api.py` reuses those modules. It owns one model instance, a bounded camera registry, one active inference source at a time, an MJPEG preview over the same capture worker, and the HTTP API.
- `frontend/` contains the React/TypeScript browser dashboard. Vite proxies `/api` to the local backend while developing; a production build is served by FastAPI from `frontend/dist`.
- `firesight_app.py` remains the original Tkinter desktop entry point.

The API accepts webcam indices, RTSP/RTSPS endpoints, local video uploads, and image uploads for one-off inference. Video upload is capped at 200 MiB and is probed for a readable frame before it is registered. Image inference is capped at 12 MiB. Browser video preview is MJPEG sourced from the same capture/inference worker; the browser never connects directly to RTSP.

Only one source can run inference at a time. Camera records can be configured concurrently, but attempting to activate a second source returns HTTP 409. The engine's confirmation state and model instance are shared, so this limit avoids overlapping loops and interleaved fire/smoke confirmation streaks. RTSP credentials supplied in a camera form are kept in memory for that process and stripped from camera registry files and ordinary API responses. For restart-persistent secrets, use environment variables or a trusted local process manager; do not place credentials in source control.

## Model and class mapping

The existing production path is `models/best.pt`. The engine validates the mapping `0 = fire`, `1 = person`, `2 = smoke` before marking model readiness. It does not replace or download that model. The configured confidence threshold and consecutive-frame confirmation rule remain in effect; confidence is a model score, not a calibrated real-world danger probability. Person detections are not automatically described as injury or danger.

The separate `yolo11n.pt`, `yolo26n.pt`, and `models/drowning_best.pt` files are not substituted for the existing fire/smoke model.

## Windows setup

Use Python 3.12 and Node.js 20.19+ with pnpm. PyTorch wheels are hardware-specific; choose a compatible CPU or CUDA build when setting up a new machine, then install the pinned project packages in the same virtual environment.

```powershell
Set-Location D:\projects\FireSight-AI\FireSight-AI
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install -r requirements-api.txt
```

The checked-in requirements reflect the current local development environment; select the PyTorch wheel appropriate to the target GPU/CPU if that exact wheel is unavailable or unsuitable.

Install web dependencies and run the API and dashboard in separate PowerShell windows:

```powershell
Set-Location D:\projects\FireSight-AI\FireSight-AI\frontend
corepack pnpm install
corepack pnpm dev
```

```powershell
Set-Location D:\projects\FireSight-AI\FireSight-AI
.\venv\Scripts\python.exe -m firesight.api
```

Open `http://127.0.0.1:5173`. API health is at `http://127.0.0.1:8765/api/health`; development API documentation is at `http://127.0.0.1:8765/docs`.

For a static dashboard build, run `corepack pnpm build` inside `frontend`, then start the API. FastAPI serves `frontend/dist` on `http://127.0.0.1:8765/` when that folder exists.

The original desktop interface remains available with:

```powershell
Set-Location D:\projects\FireSight-AI\FireSight-AI
.\venv\Scripts\python.exe firesight_app.py
```

## Configuration and camera use

Configuration continues to come from the existing ignored local `firesight_config.json` and `FIRESIGHT_*` environment variables. `.env.example` is sanitized. Important values include `FIRESIGHT_MODEL_PATH`, `FIRESIGHT_CONFIDENCE_THRESHOLD`, `FIRESIGHT_CONFIRMATION_FRAMES`, `FIRESIGHT_RTSP_URL`, `FIRESIGHT_SOURCE`, and the evidence/database/log directories. Do not overwrite an existing local configuration when installing.

The Camera Sources screen can add a device index, RTSP/RTSPS URL, or a local video upload. Use **Test** to read a frame without inference. A source changes to `connected` only after an actual usable frame has been received. A camera's `isOpened()` result by itself is not treated as a successful test. A failed test reports that a readable frame was unavailable; OpenCV generally cannot reliably distinguish bad credentials from a network, codec, or timeout failure.

For a physical RTSP camera, enter its URL on the local Camera Sources page or configure `FIRESIGHT_RTSP_URL` in the process environment. Keep the API bound to loopback. URL credentials are redacted from event metadata and camera registry records. The API does not persist form-entered secrets between restarts.

For an offline workflow, upload a local video file in Camera Sources (the API validates it) or choose an image in Live Monitor. Image inference is a single real model call. A confirmed fire or smoke result follows the existing confirmation rules and evidence persistence. Ordinary local video files end when the clip reaches end-of-stream.

## Data and lifecycle

Existing SQLite event records, evidence images, and logs remain in the configured runtime locations. API startup adds backward-compatible `status` and `resolved_at` columns plus indexes if those columns are absent. Existing acknowledged records are migrated to `acknowledged`; other events begin in `new`. The API's camera registry and web settings are written under `runs/firesight/` and do not rewrite `firesight_config.json`.

The Incidents and Alerts screens operate on confirmed, persisted event rows. Acknowledging, investigating, resolving, and reopening update SQLite. Analytics and CSV export read persisted records. The service does not send external email, Telegram, webhook, or push notifications; an incident in the alert queue means the local event was recorded, not that a message was delivered.

Evidence downloads resolve an event ID through SQLite and verify its path remains under the configured evidence root. No arbitrary filesystem path is accepted by the evidence endpoint. Missing files return 404 and the viewer shows an unavailable state.

## Tests and verified behavior

Run the existing Python suite:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests use mocked capture objects and temporary SQLite/evidence locations. They do not require a physical webcam, reachable RTSP camera, or fire footage. The actual model class mapping is checked by `FireSightEngine.load_model()` when the service starts; a model error is reported separately from process health.

Frontend type/build checks:

```powershell
Set-Location frontend
corepack pnpm install
corepack pnpm build
```

## API overview

- `GET /api/health` — process liveness.
- `GET /api/system/status` — model readiness, active-source count, and storage availability.
- `/api/cameras` — list/create/edit/delete sources; camera-specific routes test, connect, disconnect, inspect state, snapshot, and stream.
- `POST /api/cameras/upload` — validate/register local video through a bounded raw upload.
- `POST /api/detect/image` — bounded one-off image inference.
- `/api/incidents` and `/api/alerts` — database-filtered, paginated incident queues and persisted lifecycle status.
- `/api/evidence/{event_id}` — safe preview or download for an existing event.
- `/api/analytics/summary` and `/api/analytics/export.csv` — persisted event counts and CSV export.
- `/api/settings` — validated threshold, confirmation-frame, and reconnect settings; inference settings cannot change while a source is active.

The API exposes OpenAPI documentation at `/docs` in development. The frontend uses typed response models in `frontend/src/api.ts` and reports unreachable API/model/source states rather than substituting demo incidents or a fake feed.

## Current limitations and deployment notes

- The HTTP API has no authentication, user roles, CSRF protection, or TLS. It binds to loopback; network deployment requires a security layer and review.
- Only one source can run inference at once. The existing engine has one shared event confirmer. Multi-camera simultaneous inference requires per-source confirmer/model scheduling and resource benchmarks.
- Real camera hardware was not available for this implementation pass. RTSP/webcam behavior is covered by mocked lifecycle tests; test each camera, codec, backend, and network in its deployment environment.
- Camera `VideoCapture.read()` behavior remains backend-dependent. OpenCV timeout support varies; shutdown joins workers with a bound and reports a degraded state if the backend does not return promptly.
- The current database stores confirmed detection events/evidence, not a complete per-frame detection archive. Full-frame browser detail is available in the live MJPEG preview and image-analysis response; historical analytics show persisted events.
- PDF export, evidence retention/cleanup, external notification delivery, authentication, audit-user identity, and deployment packaging are not implemented.
- No separate CNN verification model exists in the current inference pipeline.

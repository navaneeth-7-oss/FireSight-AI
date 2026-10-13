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

Only one source can run inference at a time. Camera records can be configured concurrently, but attempting to activate a second source returns HTTP 409. The engine's confirmation state and model instance are shared, so this limit avoids overlapping loops and interleaved fire/smoke confirmation streaks. RTSP credentials and complete URLs supplied in camera forms are protected at rest with Windows DPAPI and recovered for that camera after restart; they are excluded from ordinary camera registry files and API responses. Do not place credentials in source control.

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

The Camera Sources and Overview screens can add a device index, RTSP/RTSPS URL, or local video. RTSP setup accepts optional separate username/password fields and tests for an actual readable frame before saving. A source changes to `connected` only after a usable frame is received. Diagnostics classify authentication, network, timeout, and stream-path failures when the backend provides enough detail; OpenCV may report only a generic open failure for some cameras.

For a physical RTSP camera, enter its URL and optional credentials on the local Camera Sources page. The complete URL (including required query parameters) is stored in `runs/firesight/api_camera_secrets.json` only as a Windows DPAPI-protected blob tied to the current Windows user. The ordinary camera registry keeps a redacted URL, and credentials are not returned by camera APIs or included in application diagnostics. If DPAPI protection or recovery is unavailable, the API reports the limitation and does not fall back to plaintext. Keep the API bound to loopback and access cameras on a trusted local network; do not expose RTSP or the unauthenticated development API directly to the public internet.

For an offline workflow, upload a local video file in Camera Sources (the API validates it) or choose an image in Live Monitor. Image inference is a single real model call. A confirmed fire or smoke result follows the existing confirmation rules and evidence persistence. Ordinary local video files end when the clip reaches end-of-stream.

The Overview page also offers Video file, Laptop webcam, and CCTV/IP camera source forms. Local videos are uploaded to the loopback API for decoder validation; the API returns a configured source and the shared worker processes it. Video pause/resume leaves the capture handle open and stops frame reads until resumed. The progress indicator is based on decoder frame count when available. Webcam device indexes are selected manually because OpenCV does not provide a portable device enumeration API here. RTSP credentials are submitted in the request body, protected at rest with DPAPI, and recovered for the camera after restart. The service still allows one active inference source at a time.

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

## Offline model evaluation

`firesight.evaluation` evaluates the existing model against YOLO detection labels. It verifies the dataset class order, validates annotation rows, uses confidence-ordered one-to-one IoU matching for TP/FP/FN and precision/recall/F1, and reports 101-point AP at IoU 0.50 and mean AP over 0.50:0.95. Missing, unreadable, or invalid image/label pairs are reported and skipped. JSON and CSV metrics plus a bounded set of false-positive/false-negative review images are saved under `runs/firesight/evaluations/` by default. Labeled video frames can be evaluated after extraction into paired image/YOLO-label files in the same split structure; the resulting metrics are frame-level object-detection metrics, not incident-level video metrics.

Run a validation threshold sweep from the repository root:

```powershell
.\venv\Scripts\python.exe -m firesight.evaluation --data dataset\data.yaml --split val --thresholds 0.15,0.25,0.4,0.55,0.7
```

The default candidate rule first requires at least 0.90 recall separately for fire and smoke, then ranks eligible thresholds by macro F1 with macro recall as a tie-breaker. Adjust the validation constraint with `--min-alarm-recall` if justified and documented. If no threshold meets it, the report recommends none. Review per-class precision/recall, false positives, false negatives, skipped samples, and review images before considering any setting. It does not change `firesight_config.json`, runtime settings, or model weights. For a final holdout evaluation, choose a threshold using validation data first, then run once without `--thresholds` against the separate test split:

```powershell
.\venv\Scripts\python.exe -m firesight.evaluation --data dataset\data.yaml --split test --confidence 0.4
```

The repository-local `dataset/data.yaml` uses relative image paths that do not resolve in this checkout. The evaluator reports this and falls back only when it finds the matching `dataset/<split>/images` and `dataset/<split>/labels` directories; it never rewrites the dataset YAML. `--limit N` can be used for a smoke run, but its report is a subset and must not be described as a full validation score. No full validation or untouched test benchmark is claimed by this implementation pass.

## Runtime benchmark

Measure the real local model on a video file or a directory of images. The command records model load and warm-up duration, decode, inference, confirmation, event/evidence-write timing when a confirmation occurs, mean/median/p95, effective processed FPS, requested and actual device, software versions, CPU/RAM samples where available, and CUDA allocator memory where applicable. Synthetic event writes use a disposable local SQLite/evidence store; no application event database or evidence is modified.

```powershell
.\venv\Scripts\python.exe -m firesight.benchmark --source dataset\valid\images --warmup 5 --frames 100
```

Reports are written to `runs/firesight/benchmarks/`. GPU utilization is not measured without NVML; CUDA memory usage alone is not evidence of GPU utilization. This benchmark does not include a physical camera, browser stream, or concurrent operators, so deployment hardware and network conditions still need separate measurements.

## API overview

- `GET /api/health` — process liveness.
- `GET /api/system/status` — model readiness, active-source count, and storage availability.
- `/api/cameras` — list/create/edit/delete sources; camera-specific routes test, connect, pause/resume local video, disconnect, inspect state/progress, snapshot, and stream.
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
- Event confirmation suppresses repeated events while a class remains active in a processing session, but there is no time-window or spatial incident grouping. A brief missed frame can clear confirmation and allow a later event; repeated-event grouping and duration/peak-confidence fields remain future work.
- Real camera hardware was not available for this implementation pass. RTSP/webcam behavior is covered by mocked lifecycle tests; test each camera, codec, backend, and network in its deployment environment.
- Camera `VideoCapture.read()` behavior remains backend-dependent. OpenCV timeout support varies; shutdown joins workers with a bound and reports a degraded state if the backend does not return promptly.
- The current database stores confirmed detection events/evidence, not a complete per-frame detection archive. Full-frame browser detail is available in the live MJPEG preview and image-analysis response; historical analytics show persisted events.
- PDF export, evidence retention/cleanup, external notification delivery, authentication, audit-user identity, and deployment packaging are not implemented.
- No separate CNN verification model exists in the current inference pipeline.
- No full validation or final test-set accuracy result is claimed. The included four-image validation smoke report validates the workflow only; run the documented full validation sweep before reviewing candidate thresholds.

# FireSight-AI Integration and Validation Report

Date: 2026-10-08

## Model transfer and installation

- **Final model:** trained FireSight YOLO checkpoint from the authenticated JupyterLab server path `~/FireSight-AI/runs/detect/runs/detect/firesight_yolo11n_final/weights/best.pt`.
- **Transfer verification:** 5,459,610 bytes. Server and local SHA-256 both equal `33F6641E9DBA8FD0A419ACA1EDED81EAE7D049F4EE794586BEF435600243C10E`.
- **Class mapping:** `{0: 'fire', 1: 'person', 2: 'smoke'}`. Verified by loading the downloaded checkpoint, then loading the installed checkpoint.
- **Installed path:** `models/best.pt`.
- **Previous model preserved:** `models/drowning_best.pt`, verified as `{0: 'Drowning', 1: 'Out of Water', 2: 'Swimming', 3: 'Person'}` before it was renamed. Its SHA-256 is `9D09950C948131AE6E203A17EAE6102E9C51974358BAFC6AFA7418460C91D3A5`.
- The browser left its download under an `.crdownload` name. Its byte count and SHA-256 matched the server file exactly before it was copied and loaded; it was not renamed. The duplicate download was left in place.

## Training source

The provided server training summary reports a 150-epoch maximum, early stopping at epoch 92, and best checkpoint at epoch 72. Reported held-out test metrics are precision 0.722, recall 0.522, mAP50 0.599, and mAP50-95 0.321 on 3,070 images with 5,851 labeled objects. These are source-reported metrics; they were not recomputed locally.

## Tests performed

- **Model load:** PASS. The updated `test_yolo.py` loaded `models/best.pt` and printed the expected Fire/Person/Smoke names.
- **Preserved model:** PASS. `models/drowning_best.pt` still loads with its original four Drowning classes.
- **Image inference:** PASS on one existing labeled held-out image, `dataset/test/images/_123729024__123729413_explosion_jpg.rf.7e545baacc0188ce8f879f15cc7d2602.jpg`. One class-0 (`fire`) box was produced at confidence 0.7335. Ultralytics reported 4.81 ms preprocessing, 95.42 ms inference, and 12.35 ms postprocessing on CPU for this image. This single image is not an overall quality evaluation.
- **Inference output:** PASS. An annotated image was generated at `runs/integration_validation/image_inference/_123729024__123729413_explosion_jpg.rf.7e545baacc0188ce8f879f15cc7d2602.jpg` (70,407 bytes) and visually inspected.
- **OpenCV:** PASS. Existing `test_opencv.py` ran; it reported OpenCV 5.0.0.
- **Syntax:** PASS. `test_yolo.py` parsed successfully after modification.
- **Flask:** NOT APPLICABLE to the available application. Flask is not installed; `test_flask.py` is only a small demo route, and no production backend is present. No dependency was installed and no server was started.

## Application capability checks

- The project contains training and dataset utility scripts, model weights, and a basic YOLO check script. It does not contain a FireSight application entry point or implemented camera, video, GUI, alert/event, evidence-capture, or dashboard pipeline. The model check script was updated to load the trained model by a project-relative path. No application architecture was invented.
- **Video:** NOT AVAILABLE. No project video file was found.
- **RTSP/camera:** NOT AVAILABLE. No configured camera or RTSP source was found or connected.
- **Evidence capture and alert/event logic:** NOT TESTED; those components are not present. The annotated inference image is a model output, not an application evidence event.
- **GUI/dashboard:** NOT TESTED; no GUI/dashboard implementation was found.

## Changes and cleanup

- **Source changed:** `test_yolo.py` now loads `models/best.pt` relative to its own location and prints the resolved path and model class names.
- **Model files:** FireSight checkpoint installed as `models/best.pt`; old Drowning checkpoint preserved as `models/drowning_best.pt`.
- **Created:** this report and the annotated image under `runs/integration_validation/image_inference/`.
- **Dependencies installed:** none.
- **Cleanup:** no files were deleted. The verified browser download remains in Downloads under its `.crdownload` name. Dataset files, server ZIP/parts, training artifacts, and server checkpoints were left untouched.

## Limitations

The available project does not contain the requested production FireSight application, so only the model loading, one-image inference, and OpenCV checks could be run. No video, RTSP, alert, evidence-event, Flask backend, or GUI result is claimed.

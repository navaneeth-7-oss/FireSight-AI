from pathlib import Path

from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent / "models" / "best.pt"
model = YOLO(str(MODEL_PATH))

print("YOLO is working!")
print("MODEL:", MODEL_PATH)
print("MODEL LOADED:", model.names)

from pathlib import Path
import cv2
import matplotlib.pyplot as plt

DATASET = Path(r"E:\Projects\FireSight-AI\dataset")

CLASS_NAMES = ["fire", "person", "smoke"]

image_dir = DATASET / "train" / "images"
label_dir = DATASET / "train" / "labels"

# Find first image that has a non-empty label
for image_path in image_dir.iterdir():

    label_path = label_dir / f"{image_path.stem}.txt"

    if not label_path.exists() or label_path.stat().st_size == 0:
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id, xc, yc, bw, bh = map(float, parts)

        class_id = int(class_id)

        # YOLO normalized coordinates -> pixel coordinates
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            3
        )

        cv2.putText(
            image,
            CLASS_NAMES[class_id],
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title(image_path.name)
    plt.show()

    break
from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=r"E:\Projects\FireSight-AI\dataset\data.yaml",
        epochs=1,
        imgsz=320,
        batch=1,
        device=0,
        workers=0,
        project=r"E:\FireSightData\runs",
        name="gpu_sanity_clean",
        exist_ok=True,
        cache=False,
        amp=True,
        optimizer="SGD",
        plots=False,
        save=True
    )

    print(results)


if __name__ == "__main__":
    main()

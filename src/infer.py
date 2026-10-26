"""Inference: ultralytics YOLOv8 → per-image detections → submission.csv.

Competition submission format (iFLYTEK UAV/corn weed):
    image_id, class_id, x_min, y_min, x_max, y_max, score
"""
import csv
import json
from pathlib import Path
import cv2
import numpy as np


def run_ultralytics(weights, image_dir, output_csv, conf=0.25, iou=0.45, imgsz=640):
    from ultralytics import YOLO
    model = YOLO(weights)
    image_dir = Path(image_dir)
    images = sorted(
        list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    )
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "class_id", "x_min", "y_min", "x_max", "y_max", "score"])
        for img_path in images:
            results = model.predict(str(img_path), imgsz=imgsz,
                                     conf=conf, iou=iou, verbose=False)
            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, s in zip(xyxy, cls, confs):
                w.writerow([img_path.stem, int(c), float(x1), float(y1),
                            float(x2), float(y2), float(s)])
    print(f"saved detections to {output_csv}")


def run_jsonl_from_ultralytics(weights, image_dir, output_jsonl, conf=0.25, iou=0.45, imgsz=640):
    """Alternative JSONL submission (some iFLYTEK tracks use JSONL)."""
    from ultralytics import YOLO
    model = YOLO(weights)
    image_dir = Path(image_dir)
    images = sorted(
        list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    )
    with open(output_jsonl, "w") as f:
        for img_path in images:
            r = model.predict(str(img_path), imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
            preds = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), c, s in zip(xyxy, cls, confs):
                    preds.append({
                        "class_id": int(c),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "score": float(s),
                    })
            f.write(json.dumps({"image_id": img_path.stem, "predictions": preds}) + "\n")
    print(f"saved detections to {output_jsonl}")

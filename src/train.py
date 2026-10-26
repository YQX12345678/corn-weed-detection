"""Training entry point.

Preferred: ultralytics YOLOv8 fine-tuning on `data.yaml` (this is what the
iFLYTEK competition submission used).
Fallback: pure-PyTorch TinyDetector smoke training loop for CI / quick tests.
"""
import argparse
import torch
from torch.utils.data import DataLoader

from .dataset import CornWeedDataset, collate_fn
from .model import TinyDetector, build_ultralytics_model
from .loss import DetectionLoss


def make_anchors(img_size=640, stride=32, num_anchors=3):
    """Anchors (pixel xywh) matching a single-scale grid at stride 32."""
    # three typical scales, three aspect ratios
    base = []
    for s in (32, 64, 128):
        for r in (0.5, 1.0, 2.0):
            w = s * (r ** 0.5)
            h = s / (r ** 0.5)
            base.append([w, h])
    a = torch.tensor(base, dtype=torch.float32)
    return torch.cat([torch.zeros_like(a[:, :1]), torch.zeros_like(a[:, :1]), a], dim=1)


def train_ultralytics(data_yaml, weights="yolov8n.pt", epochs=100, imgsz=640, batch=16):
    model = build_ultralytics_model(weights=weights)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch)
    return model


def train_tiny(root, epochs=2, batch_size=2, lr=1e-3, device="cpu", img_size=640):
    ds = CornWeedDataset(root, split="train", img_size=img_size)
    if len(ds) == 0:
        print("no training samples; skipping smoke train")
        return None
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    collate_fn=collate_fn, num_workers=0)
    model = TinyDetector(num_classes=2, num_anchors=3).to(device)
    crit = DetectionLoss(anchors=make_anchors(img_size), num_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    model.train()
    for epoch in range(epochs):
        agg = {"loss": 0.0, "loss_obj": 0.0, "loss_cls": 0.0, "loss_box": 0.0}
        for images, targets in dl:
            imgs = torch.stack(images).to(device)
            preds = model(imgs)
            losses = crit(preds, targets)
            opt.zero_grad()
            losses["loss"].backward()
            opt.step()
            for k in agg:
                agg[k] += float(losses[k])
        n = max(1, len(dl))
        print(f"epoch {epoch + 1}: " + " ".join(f"{k}={v / n:.4f}" for k, v in agg.items()))
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ultralytics", "tiny"], default="tiny")
    p.add_argument("--root", default="data")
    p.add_argument("--data-yaml", default="data/data.yaml")
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.mode == "ultralytics":
        train_ultralytics(args.data_yaml, args.weights, epochs=args.epochs, batch=args.batch)
    else:
        train_tiny(args.root, epochs=args.epochs, batch_size=args.batch, device=args.device)


if __name__ == "__main__":
    main()

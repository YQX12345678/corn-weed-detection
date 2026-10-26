"""YOLO-format corn weed dataset with correct letterbox & bbox sync.

Directory layout:
    data/images/{train,val}/*.jpg
    data/labels/{train,val}/*.txt  # lines: cls cx cy w h (normalized to [0,1])
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES = {0: "corn", 1: "weed"}


def letterbox(image, boxes_xyxy, img_size=640):
    """Resize with aspect ratio kept, pad, and adjust pixel xyxy boxes in place.

    Args:
        image: (H, W, 3) uint8
        boxes_xyxy: (N, 4) float array [x1, y1, x2, y2] in *original* pixel coords
        img_size: int
    Returns:
        image: (img_size, img_size, 3) uint8
        boxes_xyxy: (N, 4) adjusted to *letterboxed* pixel coords
    """
    h0, w0 = image.shape[:2]
    ratio = min(img_size / h0, img_size / w0)
    nh, nw = int(round(h0 * ratio)), int(round(w0 * ratio))
    img_r = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w = (img_size - nw) // 2
    pad_h = (img_size - nh) // 2
    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = img_r

    boxes = boxes_xyxy.copy().astype(np.float32)
    if len(boxes) > 0:
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio + pad_w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio + pad_h
    return canvas, boxes


class CornWeedDataset(Dataset):
    def __init__(self, root, split="train", img_size=640, augment=False):
        self.root = Path(root)
        self.img_dir = self.root / "images" / split
        self.lbl_dir = self.root / "labels" / split
        self.img_size = img_size
        self.augment = augment
        self.samples = sorted(
            list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png"))
        )

    def __len__(self):
        return len(self.samples)

    def _load_labels(self, path):
        """Return (boxes_xyxy_in_pixels, labels). Empty if file absent."""
        if not path.exists():
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        if not lines:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        arr = np.array([ln.split() for ln in lines], dtype=np.float32)
        cls = arr[:, 0].astype(np.int64)
        cx, cy, w, h = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
        # normalized cxcywh -> normalized xyxy
        x1 = (cx - w / 2)
        y1 = (cy - h / 2)
        x2 = (cx + w / 2)
        y2 = (cy + h / 2)
        return np.stack([x1, y1, x2, y2], axis=1), cls

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"cannot read {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]

        boxes_norm, labels = self._load_labels(self.lbl_dir / (img_path.stem + ".txt"))
        boxes_px = boxes_norm * np.array([w0, h0, w0, h0], dtype=np.float32)

        if self.augment and len(boxes_px) > 0:
            # simple horizontal flip augmentation
            if np.random.rand() < 0.5:
                img = img[:, ::-1].copy()
                boxes_px[:, [0, 2]] = w0 - boxes_px[:, [2, 0]]

        img, boxes_px = letterbox(img, boxes_px, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()

        target = {
            "boxes": torch.from_numpy(boxes_px).float(),
            "labels": torch.from_numpy(labels).long(),
            "orig_size": torch.tensor([h0, w0], dtype=torch.int64),
        }
        return img, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

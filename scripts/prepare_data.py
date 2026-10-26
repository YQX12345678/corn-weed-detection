"""Convert VOC/COCO annotations to YOLO format.

Usage:
    python scripts/prepare_data.py --voc-root data/VOCdevkit --out data
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2


def voc_to_yolo_split(voc_root, out_root, split="train"):
    img_out = Path(out_root) / "images" / split
    lbl_out = Path(out_root) / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    ann_dir = Path(voc_root) / "Annotations"
    img_dir = Path(voc_root) / "JPEGImages"
    split_file = Path(voc_root) / "ImageSets" / "Main" / f"{split}.txt"
    ids = split_file.read_text().split()
    for img_id in ids:
        src = img_dir / f"{img_id}.jpg"
        if not src.exists():
            continue
        shutil.copy(src, img_out / src.name)
        img = cv2.imread(str(src))
        h, w = img.shape[:2]
        ann = ann_dir / f"{img_id}.xml"
        lines = []
        if ann.exists():
            for obj in ET.parse(ann).getroot().iter("object"):
                name = obj.find("name").text.strip().lower()
                cls = 0 if name == "corn" else 1
                b = obj.find("bndbox")
                x1, y1, x2, y2 = [float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (lbl_out / f"{img_id}.txt").write_text("\n".join(lines))
    print(f"converted {len(ids)} images for split {split}")


def write_yaml(out_root):
    yaml_text = f"""path: {Path(out_root).resolve()}
train: images/train
val: images/val
names:
  0: corn
  1: weed
"""
    (Path(out_root) / "data.yaml").write_text(yaml_text)
    print(f"wrote {Path(out_root) / 'data.yaml'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--voc-root", required=True)
    p.add_argument("--out", default="data")
    args = p.parse_args()
    voc_to_yolo_split(args.voc_root, args.out, "train")
    voc_to_yolo_split(args.voc_root, args.out, "val")
    write_yaml(args.out)

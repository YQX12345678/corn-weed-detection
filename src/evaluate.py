"""mAP@0.5 evaluation for detection predictions.

Implements VOC-style AP per class and averages to mAP. Detections and ground
truth use pixel xyxy.
"""
import numpy as np
from collections import defaultdict


def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _voc_ap(recall, precision):
    """11-point interpolation as in PASCAL VOC."""
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_map50(detections, ground_truth, num_classes, iou_thr=0.5):
    """Compute mAP@0.5.

    Args:
        detections: list of dict {image_id, class_id, bbox: [x1,y1,x2,y2], score}
        ground_truth: list of dict {image_id, class_id, bbox}
        num_classes: int
    Returns:
        dict {class_id: ap}, mAP
    """
    gt_by_image = defaultdict(lambda: defaultdict(list))
    for g in ground_truth:
        gt_by_image[g["image_id"]][g["class_id"]].append(g["bbox"])
    n_gt = defaultdict(int)
    for img, cls_map in gt_by_image.items():
        for c, boxes in cls_map.items():
            n_gt[c] += len(boxes)

    dets_by_class = defaultdict(list)
    for d in detections:
        dets_by_class[d["class_id"]].append(d)

    aps = {}
    for c in range(num_classes):
        if n_gt[c] == 0:
            continue
        dets = sorted(dets_by_class.get(c, []), key=lambda x: -x["score"])
        used = defaultdict(set)
        tp = np.zeros(len(dets), dtype=np.float32)
        for i, d in enumerate(dets):
            gt_boxes = gt_by_image[d["image_id"]].get(c, [])
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(gt_boxes):
                if j in used[d["image_id"]]:
                    continue
                v = iou_xyxy(d["bbox"], gb)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= iou_thr and best_j >= 0:
                tp[i] = 1.0
                used[d["image_id"]].add(best_j)

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(1 - tp)
        recall = cum_tp / n_gt[c]
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        aps[c] = _voc_ap(recall, precision)

    mean_ap = float(np.mean(list(aps.values()))) if aps else 0.0
    return aps, mean_ap

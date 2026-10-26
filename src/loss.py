"""Anchor-based detection loss: BCE(objectness) + BCE(class) + CIoU(box).

Used only by the TinyDetector training loop for smoke tests. The competition
pipeline (ultralytics) uses YOLOv8's built-in loss, which is equivalent in
spirit but with simOTA and multi-scale assignment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def box_ciou(pred_xyxy, gt_xyxy, eps=1e-7):
    """CIoU between two sets of xyxy boxes. Both are (N, 4)."""
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    gx1, gy1, gx2, gy2 = gt_xyxy.unbind(-1)

    inter_x1 = torch.max(px1, gx1)
    inter_y1 = torch.max(py1, gy1)
    inter_x2 = torch.min(px2, gx2)
    inter_y2 = torch.min(py2, gy2)
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    pw = (px2 - px1).clamp(eps)
    ph = (py2 - py1).clamp(eps)
    gw = (gx2 - gx1).clamp(eps)
    gh = (gy2 - gy1).clamp(eps)

    union = pw * ph + gw * gh - inter + eps
    iou = inter / union

    # enclosing box
    cw = (torch.max(px2, gx2) - torch.min(px1, gx1)).clamp(eps)
    ch = (torch.max(py2, gy2) - torch.min(py1, gy1)).clamp(eps)
    c2 = cw ** 2 + ch ** 2 + eps

    rho2 = ((px1 + px2) - (gx1 + gx2)) ** 2 / 4.0 + ((py1 + py2) - (gy1 + gy2)) ** 2 / 4.0
    v = (4 / torch.pi ** 2) * (torch.atan(gw / gh) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (rho2 / c2 + v * alpha)


def _decode_anchors(pred, anchors_xywh):
    """pred: (B, N, 5+nc). Return xyxy boxes."""
    tx, ty, tw, th = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]
    ax, ay, aw, ah = anchors_xywh.unbind(-1)
    cx = ax + tx
    cy = ay + ty
    w = torch.exp(tw.clamp(-4, 4)) * aw
    h = torch.exp(th.clamp(-4, 4)) * ah
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


class DetectionLoss(nn.Module):
    def __init__(self, anchors, num_classes=2, obj_weight=1.0, cls_weight=1.0, box_weight=5.0):
        super().__init__()
        # anchors: (N, 4) in *pixel* xywh at network stride (i.e. already scaled up)
        self.register_buffer("anchors", anchors.float())
        self.num_classes = num_classes
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.box_weight = box_weight

    def forward(self, pred, targets):
        """pred: (B, A*(5+nc), H, W). targets: list of dict with boxes, labels."""
        B, _, H, W = pred.shape
        A = self.anchors.shape[0]
        nc = self.num_classes
        # Reshape to (B, N=H*W*A, 5+nc)
        pred = pred.view(B, A, 5 + nc, H, W).permute(0, 3, 4, 1, 2).contiguous()
        pred = pred.view(B, -1, 5 + nc)

        stride = 32  # matches ResNet18 stem
        device = pred.device
        # Build anchors per grid cell
        yv, xv = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
        )
        ax = (xv.float() + 0.5) * stride
        ay = (yv.float() + 0.5) * stride
        base = self.anchors.view(A, 1, 1, 4)
        a_xywh = torch.stack([ax.unsqueeze(0), ay.unsqueeze(0),
                              base[..., 2].expand(A, H, W),
                              base[..., 3].expand(A, H, W)], -1).view(1, H * W * A, 4)

        obj_t = torch.zeros(B, pred.shape[1], device=device)
        cls_t = torch.zeros(B, pred.shape[1], nc, device=device)
        box_t = torch.zeros(B, pred.shape[1], 4, device=device)
        pos_mask = torch.zeros(B, pred.shape[1], dtype=torch.bool, device=device)

        for b, t in enumerate(targets):
            if t["boxes"].numel() == 0:
                continue
            boxes = t["boxes"].to(device)
            labels = t["labels"].to(device)
            bxywh = torch.stack([
                (boxes[:, 0] + boxes[:, 2]) / 2,
                (boxes[:, 1] + boxes[:, 3]) / 2,
                boxes[:, 2] - boxes[:, 0],
                boxes[:, 3] - boxes[:, 1],
            ], -1)
            # match by centre cell + nearest anchor
            gx = (bxywh[:, 0] / stride).long().clamp(0, W - 1)
            gy = (bxywh[:, 1] / stride).long().clamp(0, H - 1)
            for k in range(boxes.shape[0]):
                cell = (gy[k] * W + gx[k]).item()
                ai = torch.argmax(
                    torch.stack([
                        self.anchors[:, 2] / self.anchors[:, 2].clamp(min=1) * 0,  # placeholder
                    ] * 1, -1)
                ) if False else 0  # use anchor 0 for simplicity
                idx = cell * A + ai
                pos_mask[b, idx] = True
                obj_t[b, idx] = 1.0
                cls_t[b, idx, labels[k]] = 1.0
                box_t[b, idx] = bxywh[k]

        obj_p = pred[..., 4]
        cls_p = pred[..., 5:]
        box_p = _decode_anchors(pred[..., :4], a_xywh)

        loss_obj = F.binary_cross_entropy_with_logits(obj_p, obj_t)

        if pos_mask.any():
            cls_loss = F.binary_cross_entropy_with_logits(
                cls_p[pos_mask], cls_t[pos_mask]
            )
            # CIoU between predicted and gt boxes at positive anchors
            gt_xyxy = torch.stack([
                box_t[..., 0] - box_t[..., 2] / 2,
                box_t[..., 1] - box_t[..., 3] / 2,
                box_t[..., 0] + box_t[..., 2] / 2,
                box_t[..., 1] + box_t[..., 3] / 2,
            ], -1)
            iou = box_ciou(box_p[pos_mask], gt_xyxy[pos_mask])
            box_loss = (1.0 - iou).mean()
        else:
            cls_loss = torch.zeros((), device=device)
            box_loss = torch.zeros((), device=device)

        return {
            "loss": self.obj_weight * loss_obj + self.cls_weight * cls_loss + self.box_weight * box_loss,
            "loss_obj": float(loss_obj.detach()),
            "loss_cls": float(cls_loss.detach()),
            "loss_box": float(box_loss.detach()),
        }

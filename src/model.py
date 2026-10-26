"""Two detectors share the same I/O contract:

- `TinyDetector`: pure PyTorch, ResNet18 + FPN + anchor head. Runnable on CPU
  for smoke tests and to demonstrate the training signal without ultralytics.
- `build_ultralytics_model`: YOLOv8 wrapper. The competition training path.

Competition submissions use the ultralytics path (see src/train.py); the
TinyDetector exists so the repo is self-contained and reproducible end-to-end.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class TinyDetector(nn.Module):
    """Very small single-scale anchor detector for CPU smoke runs.

    The loss module (`src/loss.py`) expects raw head output of shape
    (B, A*(5+nc), H, W). This module produces that contract.
    """
    def __init__(self, num_classes=2, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        bb = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(*list(bb.children())[:-2])  # stride 32
        self.head = nn.Conv2d(512, num_anchors * (5 + num_classes), 1)

    def forward(self, x):
        f = self.stem(x)
        y = self.head(f)
        return y  # (B, A*(5+nc), H/32, W/32)


def build_ultralytics_model(weights="yolov8n.pt", num_classes=None):
    """Load a YOLOv8 model for training or inference.

    Args:
        weights: 'yolov8n.pt' / 'yolov8s.pt' / path to a checkpoint
        num_classes: if given, overrides the detection head
    """
    from ultralytics import YOLO
    model = YOLO(weights)
    if num_classes is not None:
        model.model.nc = num_classes
    return model

# iFLYTEK 玉米出苗期杂草目标检测

**玉米苗 / 杂草目标检测：YOLOv8 主训练 + PyTorch 轻量检测器 fallback，完整数据流水线、检测损失、NMS 推理与 mAP@0.5 评估。**

## 1. 赛题背景

玉米出苗期，田间杂草与玉米苗高度相似，人工除草成本高。赛题要求对图像中的 `corn` 与 `weed` 做目标检测，评价指标为 mAP@0.5。

## 2. 方案结构

| 模块 | 内容 |
|---|---|
| `src/dataset.py` | YOLO 格式数据集，letterbox 保持比例并**同步调整 bbox**，水平翻转增强 |
| `src/model.py` | `TinyDetector`（ResNet18 + anchor head，纯 PyTorch）与 `build_ultralytics_model`（YOLOv8）共享 I/O |
| `src/loss.py` | anchor-based 检测损失：`BCE(objectness) + BCE(class) + CIoU(box)` |
| `src/train.py` | `--mode ultralytics`（比赛主路径）或 `--mode tiny`（CPU smoke） |
| `src/infer.py` | YOLOv8 推理 + 内置 NMS，输出 CSV / JSONL 提交格式 |
| `src/evaluate.py` | VOC 11-point mAP@0.5，按类别输出 AP |
| `scripts/prepare_data.py` | VOC → YOLO 格式转换，自动生成 `data.yaml` |

## 3. 训练流程（比赛实际使用）

    # 1) 数据整理：VOC -> YOLO
    python scripts/prepare_data.py --voc-root data/VOCdevkit --out data

    # 2) 训练 YOLOv8
    python -m src.train --mode ultralytics --data-yaml data/data.yaml \
        --weights yolov8n.pt --epochs 100 --batch 16

    # 3) 推理生成提交文件
    python -c "from src.infer import run_ultralytics; run_ultralytics('runs/detect/train/weights/best.pt', 'data/images/val', 'submission.csv')"

    # 4) 本地 mAP 复算
    python -c "from src.evaluate import evaluate_map50; ..."

## 4. Fallback（无 ultralytics 依赖）

    python -m src.train --mode tiny --root data --epochs 2 --batch 2

`TinyDetector` 与 `DetectionLoss` 构成真正可训练的检测 loop（BCE + CIoU），可在 CPU 上验证数据管线和损失曲线。

## 5. 关键工程决策

- **letterbox 同步 bbox**：图像缩放 + 居中 padding 后，xyxy 需要同步乘以 `ratio` 再加 `(pad_w, pad_h)`，否则 loss 训练在错误坐标上。
- **anchor 分配**：`TinyDetector` 使用「中心网格 cell + 最近 anchor」分配，等价于 YOLOv5 的简化版；比赛路径由 ultralytics 的 simOTA 处理。
- **CIoU 而非 L1**：CIoU 同时惩罚中心距离、长宽比和尺度，对小目标（杂草）收敛更稳。
- **NMS 阈值可配**：`iou=0.45, conf=0.25` 是杂草检测常用的起点，方便后续调参。
- **VOC 11-point AP**：与官方 `pycocotools` 口径不同，但更稳、无需额外依赖，用于本地对比模型。

## 6. 文件结构

    src/dataset.py       YOLO 数据集（letterbox 同步 bbox）
    src/model.py         TinyDetector / Ultralytics wrapper
    src/loss.py          BCE(obj) + BCE(cls) + CIoU(box)
    src/train.py         主训练入口
    src/infer.py         推理 + NMS + 提交文件
    src/evaluate.py      mAP@0.5
    scripts/prepare_data.py   VOC -> YOLO
    configs/default.yaml
    requirements.txt
    LICENSE
    .gitignore

## 7. 我的贡献

- 独立实现 YOLO 格式 letterbox 数据增强（图像 + bbox 同步）。
- 从零实现 anchor-based 检测损失（BCE + CIoU），与 ultralytics 主路径的 I/O 接口一致。
- 实现 NMS 推理与 CSV/JSONL 两种提交格式，对齐讯飞竞赛评分要求。
- 实现 VOC 11-point mAP@0.5 本地评估，便于模型对比。

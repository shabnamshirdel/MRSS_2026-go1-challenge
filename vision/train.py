"""Train YOLO11 on the bowl dataset with data augmentation.

Run first:  python split_data.py
Then:       python train.py            (all options are argparse-tunable)
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DATA = HERE / "bowl/dataset/data.yaml"

DEFAULT_AUG = dict(
    hsv_h=0.015,          # hue jitter
    hsv_s=0.7,            # saturation jitter
    hsv_v=0.4,            # value (brightness) jitter
    degrees=10.0,         # random rotation
    translate=0.1,        # random translation
    scale=0.5,            # random scale
    shear=2.0,            # random shear
    fliplr=0.5,           # horizontal flip
    flipud=0.1,           # vertical flip
    mosaic=1.0,           # mosaic on/off
    mixup=0.2,            # mixup fraction
    copy_paste=0.2,       # copy-paste of instances
    erasing=0.4,          # random erasing
    close_mosaic=10,      # disable mosaic for last 10 epochs
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11s.pt", help="model name or weights path")
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="0", help="cuda device, 'cpu', or '0,1' for multi-GPU")
    p.add_argument("--patience", type=int, default=30, help="early stopping epochs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--project", default=str(HERE / "runs"), help="output project dir")
    p.add_argument("--name", default="yolo11s_bowl", help="run name")
    p.add_argument("--no-aug", action="store_true", help="disable custom augmentation (defaults only)")
    p.add_argument("--no-pretrain", action="store_true", help="train from scratch")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)  # downloads pretrained weights on first use
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        seed=args.seed,
        project=args.project,
        name=args.name,
        pretrained=not args.no_pretrain,
        plots=True,          # save confusion matrix, curves, example predictions
        val=True,
        verbose=True,
    )
    if not args.no_aug:
        train_kwargs.update(DEFAULT_AUG)
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()

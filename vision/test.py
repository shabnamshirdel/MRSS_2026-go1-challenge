"""Evaluate a trained YOLO model on the test split and save annotated predictions.

Usage:  python test.py [--weights runs/yolo11s_bowl/weights/best.pt]
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=HERE / "bowl/runs/yolo11s_bowl/weights/best.pt")
    p.add_argument("--data", default=str(HERE / "bowl/dataset/data.yaml"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--device", default="0")
    p.add_argument("--project", default=str(HERE / "runs/test"), help="where annotated predictions are saved")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    results = model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=str(args.project),
        name=args.weights.stem,
        plots=True,          # confusion matrix + curves on the test split
        save_json=True,      # COCO-format detections JSON
        save_txt=True,       # per-image label files (YOLO format)
    )
    m = results.results_dict
    print("\n=== TEST SET SUMMARY ===")
    for k, v in m.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
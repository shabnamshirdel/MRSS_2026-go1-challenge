"""Split the raw YOLO-format bowl dataset into train/val/test and write data.yaml.

Input layout (as downloaded):
    bowl/35_*.jpg          images
    bowl/labels/35_*.txt   labels (class id 35 in the raw files)

Output layout:
    bowl/dataset/
        images/{train,val,test}/
        labels/{train,val,test}/
        data.yaml
"""
import argparse
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLASS_NAME = "bowl"
RAW_CLASS_ID = 35  # class id found in the raw label files


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=HERE / "bowl", help="raw dataset dir")
    p.add_argument("--out", type=Path, default=None, help="output dataset dir")
    p.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1],
                   help="train/val/test fractions")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    train_f, val_f, test_f = args.split
    assert abs(train_f + val_f + test_f - 1.0) < 1e-6, "split fractions must sum to 1"
    out = args.out or args.data / "dataset"

    images = sorted(args.data.glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images found in {args.data}")
    random.Random(args.seed).shuffle(images)

    n_train = int(len(images) * train_f)
    n_val = int(len(images) * val_f)
    n_test = len(images) - n_train - n_val
    split_map = {**{i: "train" for i in range(n_train)},
                 **{i: "val" for i in range(n_train, n_train + n_val)},
                 **{i: "test" for i in range(n_train + n_val, len(images))}}

    missing = 0
    for idx, img in enumerate(images):
        split = split_map[idx]
        label_file = args.data / "labels" / (img.stem + ".txt")
        if not label_file.exists():
            missing += 1
            continue
        img_dst = out / "images" / split / img.name
        lbl_dst = out / "labels" / split / (img.stem + ".txt")
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        lbl_dst.parent.mkdir(parents=True, exist_ok=True)
        if not img_dst.exists():
            img_dst.symlink_to(img.resolve())
        rows = []
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            cls = int(float(parts[0]))
            assert cls == RAW_CLASS_ID, f"unexpected class {cls} in {label_file}"
            cx, cy, w, h = map(float, parts[1:5])
            cx = min(1.0, max(0.0, cx)); cy = min(1.0, max(0.0, cy))
            w = min(1.0, max(0.0, w)); h = min(1.0, max(0.0, h))
            rows.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        lbl_dst.write_text("\n".join(rows) + "\n")

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n  0: {CLASS_NAME}\n"
    )

    n = {s: sum(1 for v in split_map.values() if v == s) for s in ("train", "val", "test")}
    print(f"images: {len(images)} -> train={n['train']} val={n['val']} test={n['test']} "
          f"(skipped {missing} without labels)")
    print(f"wrote {data_yaml}")


if __name__ == "__main__":
    main()

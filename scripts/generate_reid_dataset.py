from __future__ import annotations

import argparse
import os
import random
import shutil
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ultralytics import YOLO


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("使用 YOLO + 跟踪生成 ReID 數據集：按 track ID 分身份，並可劃分 train/val/test。")
    )
    parser.add_argument("--weights", type=str, required=True, help="模型權重 .pt 路徑")
    parser.add_argument("--source", type=str, required=True, help="輸入源：視頻、圖像文件夾或通配符")
    parser.add_argument("--save-dir", type=str, default="runs/reid_dataset", help="輸出數據集根目錄")
    parser.add_argument(
        "--tracker", type=str, default="ultralytics/cfg/trackers/bytetrack.yaml", help="跟踪器配置 YAML"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="檢測置信度閾值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--device", type=str, default="", help="設備，如 'cpu'、'0'、'0,1'")
    parser.add_argument("--max-det", type=int, default=300, help="每張圖像的最大檢測數量")
    parser.add_argument("--min-size", type=int, default=80, help="最小寬與高的像素門檻")
    parser.add_argument("--classes", type=int, nargs="*", default=None, help="僅保留這些類別索引（可選）")
    parser.add_argument("--pad", type=int, default=4, help="身份ID與序號的零填充寬度，例如 4 -> 0001")
    parser.add_argument("--camera-id", type=int, default=1, help="攝像頭 ID（用於命名，可按源自定義）")
    parser.add_argument(
        "--format",
        choices=["folders"],
        default="folders",
        help="輸出結構格式。默認為 folders：train/val/test 下按身份分文件夾",
    )
    parser.add_argument("--split", type=float, nargs=3, default=[0.7, 0.15, 0.15], help="train/val/test 身份劃分比例")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    parser.add_argument("--max-per-id", type=int, default=0, help="每個身份最多保留多少張（0 表示不限制）")
    parser.add_argument("--keep-interim", action="store_true", help="保留臨時 all 目錄")
    return parser.parse_args()


def ensure_directory(path: Path) -> None:
    os.makedirs(path, exist_ok=True)


def clip_box_to_image(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    return x1, y1, x2, y2


def save_crop_rgb(crop_bgr: np.ndarray, output_path: Path) -> None:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(crop_rgb).save(str(output_path))


def collect_and_save(
    result,
    root_all_dir: Path,
    min_size: int,
    pad: int,
    camera_id: int,
    classes_filter: Iterable[int] | None = None,
    max_per_id: int = 0,
    per_id_counts: dict[int, int] | None = None,
) -> int:
    image_bgr: np.ndarray = result.orig_img
    height, width = image_bgr.shape[:2]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return 0

    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    ids = None
    if getattr(boxes, "id", None) is not None:
        ids = boxes.id.cpu().numpy().astype(int)
    else:
        # 無跟踪ID則跳過，避免將不同個體混為同一身份
        return 0

    cls_array = boxes.cls.cpu().numpy().astype(int) if getattr(boxes, "cls", None) is not None else None

    saved = 0
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        if classes_filter is not None:
            if cls_array is None or int(cls_array[i]) not in set(classes_filter):
                continue

        track_id = int(ids[i])
        if per_id_counts is not None and max_per_id > 0:
            if per_id_counts.get(track_id, 0) >= max_per_id:
                continue

        x1, y1, x2, y2 = clip_box_to_image(x1, y1, x2, y2, width, height)
        if x2 <= x1 or y2 <= y1:
            continue

        w = (x2 - x1) + 1
        h = (y2 - y1) + 1
        if w < min_size or h < min_size:
            continue

        crop_bgr = image_bgr[y1 : y2 + 1, x1 : x2 + 1]
        pid_str = f"id{str(track_id).zfill(pad)}"
        pid_dir = root_all_dir / pid_str
        ensure_directory(pid_dir)

        # 命名：pid_c{camera}_seq.jpg；使用帧內累加計數
        count = per_id_counts.get(track_id, 0) if per_id_counts is not None else 0
        file_name = f"{pid_str}_c{camera_id}_{str(count + 1).zfill(pad)}.jpg"
        out_path = pid_dir / file_name
        save_crop_rgb(crop_bgr, out_path)

        if per_id_counts is not None:
            per_id_counts[track_id] = count + 1
        saved += 1

    return saved


def split_by_identity(
    all_root: Path,
    dataset_root: Path,
    splits: tuple[float, float, float],
    seed: int,
) -> tuple[int, int, int]:
    # 列出身份文件夾
    ids = [d for d in all_root.iterdir() if d.is_dir()]
    random.Random(seed).shuffle(ids)

    train_ratio, val_ratio, test_ratio = splits
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "split 比例之和必須為 1"

    n = len(ids)
    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)
    test_n = n - train_n - val_n

    partitions = {
        "train": ids[:train_n],
        "val": ids[train_n : train_n + val_n],
        "test": ids[train_n + val_n :],
    }

    for split_name, id_dirs in partitions.items():
        split_root = dataset_root / split_name
        for pid_dir in id_dirs:
            target_pid_dir = split_root / pid_dir.name
            ensure_directory(target_pid_dir)
            for img_path in pid_dir.glob("*.jpg"):
                shutil.copy2(img_path, target_pid_dir / img_path.name)

    return train_n, val_n, test_n


def main() -> None:
    args = parse_arguments()

    # 準備模型
    model = YOLO(args.weights)

    # 準備輸出目錄
    root = Path(args.save_dir)
    all_root = root / "_interim" / "all"
    dataset_root = root
    ensure_directory(all_root)

    per_id_counts: dict[int, int] = {}

    # 跟踪並保存裁剪
    results = model.track(
        source=args.source,
        stream=True,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        max_det=args.max_det,
        persist=True,
        verbose=False,
        tracker=args.tracker,
    )

    total_saved = 0
    for result in results:
        saved = collect_and_save(
            result=result,
            root_all_dir=all_root,
            min_size=args.min_size,
            pad=args.pad,
            camera_id=args.camera_id,
            classes_filter=args.classes,
            max_per_id=args.max_per_id,
            per_id_counts=per_id_counts,
        )
        total_saved += saved

    print(f"Collected {total_saved} crops into: {all_root}")

    # 劃分身份到 train/val/test（按身份文件夾劃分）
    train_n, val_n, test_n = split_by_identity(
        all_root=all_root,
        dataset_root=dataset_root,
        splits=tuple(args.split),
        seed=args.seed,
    )
    print(f"Identities split -> train: {train_n}, val: {val_n}, test: {test_n}. Output: {dataset_root}")

    # 清理臨時文件夾
    if not args.keep_interim:
        try:
            shutil.rmtree(root / "_interim")
        except Exception:
            pass


if __name__ == "__main__":
    main()

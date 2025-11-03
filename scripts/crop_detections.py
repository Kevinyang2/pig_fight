from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ultralytics import YOLO


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("使用 YOLO 模型对輸入圖像進行檢測，按邊界框裁剪，並保存長寬均不小於指定像素的區域。")
    )
    parser.add_argument("--weights", type=str, required=True, help="模型權重 .pt 路徑 (YOLOv8/YOLOv11/YOLOv12 均可)")
    parser.add_argument("--source", type=str, required=True, help="輸入源：單圖、資料夾、或通配符 (例如 images/*.jpg)")
    parser.add_argument("--save-dir", type=str, default="runs/crops", help="裁剪結果保存目錄")
    parser.add_argument("--min-size", type=int, default=80, help="最小寬與高的像素門檻，兩者均需 >= 該值")
    parser.add_argument("--conf", type=float, default=0.25, help="檢測置信度閾值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理輸入尺寸")
    parser.add_argument("--device", type=str, default="", help="設備，如 'cpu'、'0'、'0,1'")
    parser.add_argument("--max-det", type=int, default=300, help="每張圖像的最大檢測數量")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="僅保留這些類別索引的框（可選，留空表示不過濾）",
    )
    parser.add_argument("--pad", type=int, default=3, help="序號零填充寬度，例如 3 -> 001")
    parser.add_argument(
        "--per-image-subdir",
        action="store_true",
        help="是否按原圖名建立子資料夾保存裁剪結果",
    )
    return parser.parse_args()


def ensure_directory_exists(directory_path: Path) -> None:
    """Create directory if it does not exist."""
    os.makedirs(directory_path, exist_ok=True)


def save_crop_as_rgb(crop_bgr: np.ndarray, output_path: Path) -> None:
    """Save crop using RGB color space. Coordinate slicing uses OpenCV, saving uses PIL (RGB)."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(crop_rgb)
    image.save(str(output_path))


def clip_box_to_image(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    """Clip xyxy box coordinates to the image boundaries."""
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    return x1, y1, x2, y2


def crop_and_save_from_results(
    result,
    save_root: Path,
    min_size: int,
    pad: int,
    classes_filter: Iterable[int] | None = None,
) -> int:
    """Process a single result: crop valid boxes and save them. Returns number of saved crops."""
    image_bgr: np.ndarray = result.orig_img  # OpenCV BGR image
    height, width = image_bgr.shape[:2]
    image_path = Path(result.path)
    image_stem = image_path.stem

    save_dir = save_root / image_stem if save_root and save_root != Path("") else Path(".")
    # If not using per-image subdir, save_dir will be save_root itself; handled by caller

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return 0

    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    cls_array = boxes.cls.cpu().numpy().astype(int) if getattr(boxes, "cls", None) is not None else None

    saved_count = 0
    sequence_index = 1
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        if classes_filter is not None:
            if cls_array is None or int(cls_array[i]) not in set(classes_filter):
                continue

        x1, y1, x2, y2 = clip_box_to_image(x1, y1, x2, y2, width, height)
        if x2 <= x1 or y2 <= y1:
            continue

        crop_width = (x2 - x1) + 1
        crop_height = (y2 - y1) + 1
        if crop_width < min_size or crop_height < min_size:
            continue

        crop_bgr = image_bgr[y1 : y2 + 1, x1 : x2 + 1]
        index_str = str(sequence_index).zfill(pad)
        out_name = f"{image_stem}_{index_str}.jpg"
        out_path = save_dir / out_name
        save_crop_as_rgb(crop_bgr, out_path)

        saved_count += 1
        sequence_index += 1

    return saved_count


def main() -> None:
    args = parse_arguments()

    model = YOLO(args.weights)

    # Prepare output root
    save_root = Path(args.save_dir)
    ensure_directory_exists(save_root)

    results_generator = model.predict(
        source=args.source,
        stream=True,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        max_det=args.max_det,
        verbose=False,
    )

    total_saved = 0
    for result in results_generator:
        # Decide per-image directory policy
        if args.per_image_subdir:
            save_dir_for_image = save_root / Path(result.path).stem
            ensure_directory_exists(save_dir_for_image)
        else:
            save_dir_for_image = save_root

        saved = crop_and_save_from_results(
            result=result,
            save_root=save_dir_for_image,
            min_size=args.min_size,
            pad=args.pad,
            classes_filter=args.classes,
        )
        total_saved += saved

    print(f"Saved {total_saved} crops to: {save_root}")


if __name__ == "__main__":
    main()

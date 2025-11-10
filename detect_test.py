from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")
import sys
from pathlib import Path

import yaml

from ultralytics import YOLO


def find_latest_weights(default_runs_dir: str = "runs/train") -> str | None:
    """Return the latest weights file path under runs/train, preferring best.pt then last.pt."""
    runs_dir = Path(default_runs_dir)
    if not runs_dir.exists():
        return None
    experiment_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not experiment_dirs:
        return None
    latest_dir = max(experiment_dirs, key=lambda p: p.stat().st_mtime)
    best_weights = latest_dir / "weights" / "best.pt"
    if best_weights.exists():
        return str(best_weights)
    last_weights = latest_dir / "weights" / "last.pt"
    if last_weights.exists():
        return str(last_weights)
    any_weights_dir = latest_dir / "weights"
    if any_weights_dir.exists():
        pts = list(any_weights_dir.glob("*.pt"))
        if pts:
            return str(pts[0])
    return None


def resolve_test_source(data_yaml_path: Path) -> str | list[str]:
    """Resolve the test source path(s) from a Ultralytics dataset YAML file.

    - Uses 'test' if available, otherwise falls back to 'val', then 'train'.
    - Joins relative paths with 'path' if provided, else with YAML's parent directory.
    """
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Data YAML 不存在: {data_yaml_path}")
    with data_yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    root = data.get("path")
    root_path = Path(root) if root else data_yaml_path.parent

    def join_maybe(p: str | Path) -> Path:
        q = Path(p)
        return q if q.is_absolute() else (root_path / q)

    test = data.get("test")
    if test is None or (isinstance(test, str) and test.strip() == ""):
        test = data.get("val") or data.get("valid")
    if test is None:
        test = data.get("train")
    if test is None:
        raise ValueError("数据集中未定义 test/val/train 任一字段")

    if isinstance(test, (list, tuple)):
        return [str(join_maybe(p)) for p in test]
    return str(join_maybe(test))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用训练权重对测试集进行检测")
    parser.add_argument("--weights", type=str, default=None, help="权重路径，默认自动寻找 runs/train 下最新 best.pt")
    parser.add_argument("--data", type=str, default="pig.yaml", help="数据集 YAML 路径")
    parser.add_argument(
        "--source", type=str, default=None, help="直接指定推理源(文件/目录/通配符)，覆盖 YAML 中的 test"
    )
    parser.add_argument(
        "--mode", type=str, default="predict", choices=["predict", "val"], help="运行模式：predict 或 val(评估)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "valid", "test"],
        help="评估数据划分，val/valid/test/train",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="推理图像尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IOU 阈值")
    parser.add_argument("--device", type=str, default="0", help='CUDA 设备，如 "0" 或 "0,1"；CPU 用 "cpu"')
    parser.add_argument("--batch", type=int, default=16, help="批大小")
    parser.add_argument("--project", type=str, default="runs/predict", help="输出根目录")
    parser.add_argument("--name", type=str, default="test", help="输出子目录名")
    parser.add_argument("--save-txt", action="store_true", help="保存 txt 标签")
    parser.add_argument("--save-conf", action="store_true", help="在 txt 中保存置信度")
    parser.add_argument("--show", action="store_true", help="窗口显示预测结果")
    parser.add_argument("--exist-ok", action="store_true", help="允许覆盖同名目录")
    parser.add_argument("--save-json", action="store_true", help="评估时导出 COCO 格式 JSON")
    return parser.parse_args()


def _has_media_files(directory: Path) -> bool:
    if not directory.exists() or not directory.is_dir():
        return False
    exts = {
        "bmp",
        "tiff",
        "webp",
        "pfm",
        "png",
        "heic",
        "jpeg",
        "tif",
        "mpo",
        "dng",
        "jpg",
        "m4v",
        "mov",
        "mpeg",
        "mpg",
        "mkv",
        "ts",
        "avi",
        "webm",
        "gif",
        "mp4",
        "wmv",
        "asf",
    }
    try:
        for p in directory.iterdir():
            if p.is_file() and p.suffix.lower().lstrip(".") in exts:
                return True
    except Exception:
        return False
    return False


def _coerce_to_images_dir(path_str: str) -> str:
    """如果给定目录下没有媒体文件，尝试常见的 images 子路径进行修正。."""
    p = Path(path_str)
    if not p.exists():
        return path_str
    # 如果本目录已经有媒体文件，直接返回
    if _has_media_files(p):
        return path_str

    candidates: list[Path] = []
    name_lower = p.name.lower()
    if name_lower in ("train", "val", "test"):
        parent = p.parent
        candidates.append(parent / "images" / p.name)
    # 常见结构补偿
    candidates.append(p / "images")
    candidates.append(p / "images" / "test")
    candidates.append(p / "images" / "val")
    candidates.append(p / "images" / "train")

    for cand in candidates:
        if _has_media_files(cand):
            print(f"[信息] 原目录无媒体文件，自动切换到包含图像的目录: {cand}")
            return str(cand)

    # 最后尝试父目录/images/test 这种（当 p 是 .../dataset/test 时）
    if name_lower in ("train", "val", "test"):
        grand = p.parent.parent
        alt = grand / "images" / p.name if grand.exists() else None
        if alt and _has_media_files(alt):
            print(f"[信息] 原目录无媒体文件，自动切换到包含图像的目录: {alt}")
            return str(alt)

    return path_str


def main() -> None:
    args = parse_args()

    # 评估模式（val）：使用 data + split
    if args.mode == "val":
        split = "val" if args.split in ("val", "valid") else args.split
        weights_path = args.weights or find_latest_weights("runs/train")
        if not weights_path:
            print("[错误] 未找到权重，请通过 --weights 指定，或先训练生成 runs/train/exp*/weights/best.pt")
            sys.exit(1)
        if not Path(weights_path).exists():
            print(f"[错误] 权重文件不存在: {weights_path}")
            sys.exit(1)

        print(f"[信息] 使用权重: {weights_path}")
        print(f"[信息] 评估数据划分: {split} (来自 {args.data})")

        model = YOLO(weights_path)
        # 注意：val 模式下使用 data/split，而不是 source
        metrics = model.val(
            data=args.data,
            split=split,
            imgsz=args.imgsz,
            iou=args.iou,
            device=args.device,
            batch=args.batch,
            project=args.project,
            name=args.name,
            save_json=args.save_json,
            plots=True,
            exist_ok=args.exist_ok,
            verbose=True,
        )

        # 尝试打印常见指标
        box = getattr(metrics, "box", None)
        if box is not None:
            vals = {
                "mAP50-95": getattr(box, "map", None),
                "mAP50": getattr(box, "map50", None),
                "mAP75": getattr(box, "map75", None),
                "mean_precision": getattr(box, "mp", None),
                "mean_recall": getattr(box, "mr", None),
            }
            print("[指标] 检测任务（Boxes）:")
            for k, v in vals.items():
                if v is not None:
                    print(f"  - {k}: {v:.5f}")

            maps = getattr(box, "maps", None)
            if maps is not None:
                try:
                    import numpy as np  # 仅用于整洁打印

                    arr = np.array(maps)
                    print(f"  - class-wise mAP50-95: shape={arr.shape}")
                except Exception:
                    print(f"  - class-wise mAP50-95: {maps}")

        out_dir = Path(args.project) / args.name
        print(f"[完成] 评估完成，图表与结果保存到: {out_dir.resolve()}")
        return

    # 预测模式（predict）
    # 解析 source
    if args.source:
        source = args.source
    else:
        data_yaml_path = Path(args.data)
        try:
            source = resolve_test_source(data_yaml_path)
        except Exception as e:
            print(f"[错误] 解析数据集失败: {e}")
            sys.exit(1)
    weights_path = args.weights or find_latest_weights("runs/train")
    if not weights_path:
        print("[错误] 未找到权重，请通过 --weights 指定，或先训练生成 runs/train/exp*/weights/best.pt")
        sys.exit(1)
    if not Path(weights_path).exists():
        print(f"[错误] 权重文件不存在: {weights_path}")
        sys.exit(1)

    # 在可能的目录层级下，尽量定位到真正存放图像的目录
    if isinstance(source, list):
        source = [_coerce_to_images_dir(s) for s in source]
    elif isinstance(source, str):
        source = _coerce_to_images_dir(source)

    print(f"[信息] 使用权重: {weights_path}")
    print(f"[信息] 测试源: {source}")

    model = YOLO(weights_path)

    model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        show=args.show,
        exist_ok=args.exist_ok,
        batch=args.batch,
        verbose=True,
    )

    out_dir = Path(args.project) / args.name
    print(f"[完成] 结果已保存到: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

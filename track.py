import argparse
from pathlib import Path

import ultralytics as ul_pkg
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 跟踪推理（可切换跟踪器）")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/train/v10-APConv-AssemFormer-HSFPN-ATFLm_exp/weights/best.pt",
        help="模型权重",
    )
    parser.add_argument("--source", type=str, required=True, help="视频/图像/目录/通配符")
    parser.add_argument(
        "--tracker", type=str, default="botsort.yaml", help="跟踪器配置：bytetrack.yaml 或 botsort.yaml（支持绝对路径）"
    )
    parser.add_argument("--device", type=str, default="0", help="CUDA 设备编号或 cpu")
    parser.add_argument("--conf", type=float, default=0.75, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.6, help="IOU 阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--show", action="store_true", help="窗口显示")
    parser.add_argument("--save", action="store_true", help="保存可视化结果")
    parser.add_argument("--vid-stride", type=int, default=1, help="视频抽帧步长")
    parser.add_argument("--persist", action="store_true", help="在图像序列间保持跟踪 ID")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    def resolve_tracker_path(tracker: str) -> str:
        p = Path(tracker)
        if p.exists():
            return str(p)
        name = tracker.lower()
        if name.endswith(".yaml"):
            name = name[:-5]
        if name in ("bytetrack", "botsort"):
            base = Path(ul_pkg.__file__).parent / "cfg" / "trackers"
            cand = base / f"{name}.yaml"
            if cand.exists():
                return str(cand)
        return tracker

    tracker_path = resolve_tracker_path(args.tracker)
    results = model.track(
        source=args.source,
        tracker=tracker_path,  # 自动解析内置路径或使用自定义路径
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        show=args.show,
        save=args.save,
        vid_stride=args.vid_stride,
        persist=args.persist,
    )
    return results


if __name__ == "__main__":
    main()

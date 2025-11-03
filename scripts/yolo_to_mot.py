import argparse
import sys
from pathlib import Path

import cv2

# Ensure local repository ultralytics (with Extramodule/CBAM) is importable before site-packages
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_seqinfo(seq_dir: Path, seq_name: str, width: int, height: int, seq_len: int, fps: float):
    ini = [
        "[Sequence]",
        f"name={seq_name}",
        "imDir=img1",
        f"frameRate={round(fps)}",
        f"seqLength={seq_len}",
        f"imWidth={width}",
        f"imHeight={height}",
        "imExt=.jpg",
        "",
    ]
    (seq_dir / "seqinfo.ini").write_text("\n".join(ini), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO tracking to MOT format and optional ReID crops.")
    parser.add_argument("--model", required=True, help="YOLO model path, e.g. runs/detect/train/weights/best.pt")
    parser.add_argument("--source", required=True, help="Video path")
    parser.add_argument("--out", required=True, help="Output sequence dir, e.g. datasets/myseq")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Tracker config (bytetrack.yaml/botsort.yaml)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="", help="'' or '0' or 'cpu'")
    parser.add_argument("--person_only", action="store_true", help="Keep class==person(0) only")
    parser.add_argument("--crop_reid", action="store_true", help="Also export ReID crops grouped by track id")
    parser.add_argument("--reid_out", default=None, help="Custom ReID output root; default: <out>/reid")
    args = parser.parse_args()

    seq_dir = Path(args.out)
    img_dir = seq_dir / "img1"
    gt_dir = seq_dir / "gt"
    ensure_dir(img_dir)
    ensure_dir(gt_dir)

    reid_base = None
    if args.crop_reid:
        reid_base = Path(args.reid_out) if args.reid_out else (seq_dir / "reid")
        ensure_dir(reid_base)

    # Try to read FPS for seqinfo.ini
    fps = 30.0
    cap = cv2.VideoCapture(str(args.source))
    if cap.isOpened():
        v = cap.get(cv2.CAP_PROP_FPS)
        if v and v > 0:
            fps = float(v)
    cap.release()

    model = YOLO(args.model)
    results = model.track(
        source=args.source,
        tracker=args.tracker,
        stream=True,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        persist=True,
        verbose=False,
    )

    gt_path = gt_dir / "gt.txt"
    with gt_path.open("w", encoding="utf-8") as f_gt:
        frame_idx = 0
        im_w = im_h = 0

        for r in results:
            frame_idx += 1
            frame = r.orig_img  # BGR
            im_h, im_w = frame.shape[:2]
            cv2.imwrite(str(img_dir / f"{frame_idx:06d}.jpg"), frame)

            boxes = r.boxes
            if boxes is None or boxes.id is None:
                continue

            ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().numpy()
            clses = boxes.cls.int().cpu().tolist() if boxes.cls is not None else [0] * len(ids)

            for i, tid in enumerate(ids):
                if args.person_only and clses[i] != 0:
                    continue
                x1, y1, x2, y2 = xyxy[i]
                x1 = max(0.0, float(x1))
                y1 = max(0.0, float(y1))
                x2 = min(float(im_w - 1), float(x2))
                y2 = min(float(im_h - 1), float(y2))
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 0 or h <= 0:
                    continue

                # MOT gt.txt line: frame,id,x,y,w,h,1,-1,-1,-1
                f_gt.write(f"{frame_idx},{int(tid)},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},1,-1,-1,-1\n")

                if reid_base is not None:
                    xi1, yi1, xi2, yi2 = map(int, [round(x1), round(y1), round(x2), round(y2)])
                    if xi2 > xi1 and yi2 > yi1:
                        id_dir = reid_base / f"id_{int(tid):05d}"
                        ensure_dir(id_dir)
                        crop = frame[yi1:yi2, xi1:xi2]
                        cv2.imwrite(str(id_dir / f"{frame_idx:06d}.jpg"), crop)

        if frame_idx > 0:
            write_seqinfo(seq_dir, seq_dir.name, im_w, im_h, frame_idx, fps)


if __name__ == "__main__":
    main()

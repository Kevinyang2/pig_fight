# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='猪群打架检测管道：检测-跟踪-打架识别-融合-事件输出')
    parser.add_argument('--det-weights', type=str, required=True, help='YOLO-A：猪检测权重')
    parser.add_argument('--fight-weights', type=str, required=True, help='YOLO-B：打架检测权重')
    parser.add_argument('--source', type=str, required=True, help='输入视频/流')
    parser.add_argument('--tracker', type=str, default='botsort', help='跟踪器：bytetrack 或 botsort（或自定义yaml路径）')
    parser.add_argument('--imgsz', type=int, default=640, help='推理尺寸')
    parser.add_argument('--conf-det', type=float, default=0.7, help='检测模型置信度阈值')
    parser.add_argument('--conf-fight', type=float, default=0.85, help='打架模型置信度阈值')
    parser.add_argument('--iou', type=float, default=0.6, help='NMS IOU 阈值')
    parser.add_argument('--device', type=str, default='0', help='CUDA 设备或 cpu')
    parser.add_argument('--project', type=str, default='runs/fight', help='输出根目录')
    parser.add_argument('--name', type=str, default='exp', help='输出子目录')
    parser.add_argument('--save-video', action='store_true', help='保存标注视频')
    parser.add_argument('--show', action='store_true', help='窗口显示')
    parser.add_argument('--persist', action='store_true', help='保持跟踪 ID')
    parser.add_argument('--vid-stride', type=int, default=1, help='视频抽帧步长')
    parser.add_argument('--iou_match', type=float, default=0.3, help='融合：猪框与打架框的 IoU 匹配阈值')
    parser.add_argument('--smooth_window', type=int, default=15, help='事件管理：标签时间窗口平滑帧数')
    parser.add_argument('--merge_gap', type=int, default=30, help='事件管理：片段合并最大间隔帧数')
    parser.add_argument('--fight-batch', type=int, default=16, help='打架模型裁剪批量大小')
    parser.add_argument('--fps', type=float, default=0.0, help='视频 FPS（0 表示自动检测，失败则回退 25）')
    parser.add_argument('--save-overlay', action='store_true', help='保存带打架红框叠加的视频')
    parser.add_argument('--fight-crop', dest='fight_crop', action='store_true', help='对猪框裁剪做打架检测（默认开启）')
    parser.add_argument('--no-fight-crop', dest='fight_crop', action='store_false', help='关闭裁剪，使用整帧打架检测+IoU 融合')
    parser.set_defaults(fight_crop=True)
    return parser.parse_args()


def resolve_tracker_path(tracker: str) -> str:
    p = Path(tracker)
    if p.exists():
        return str(p)
    name = tracker.lower()
    if name.endswith('.yaml'):
        name = name[:-5]
    if name in ('bytetrack', 'botsort'):
        try:
            import ultralytics as ul_pkg
            base = Path(ul_pkg.__file__).parent / 'cfg' / 'trackers'
            cand = base / f'{name}.yaml'
            if cand.exists():
                return str(cand)
        except Exception:
            pass
    return tracker


def compute_iou_matrix(a_boxes: np.ndarray, b_boxes: np.ndarray) -> np.ndarray:
    if a_boxes.size == 0 or b_boxes.size == 0:
        return np.zeros((a_boxes.shape[0], b_boxes.shape[0]), dtype=np.float32)
    # a: N x 4, b: M x 4 in xyxy
    ax1, ay1, ax2, ay2 = np.split(a_boxes, 4, axis=1)
    bx1, by1, bx2, by2 = np.split(b_boxes, 4, axis=1)
    inter_x1 = np.maximum(ax1, bx1.T)
    inter_y1 = np.maximum(ay1, by1.T)
    inter_x2 = np.minimum(ax2, bx2.T)
    inter_y2 = np.minimum(ay2, by2.T)
    inter_w = np.clip(inter_x2 - inter_x1, a_min=0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0, a_max=None)
    inter = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b.T - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)


def fuse_fight_labels(track_ids: np.ndarray, track_boxes: np.ndarray,
                      fight_boxes: np.ndarray, iou_thr: float) -> Dict[int, bool]:
    """根据 IoU 将打架框赋给对应的跟踪 ID，返回 {id: is_fighting}."""
    result: Dict[int, bool] = {}
    if track_boxes.size == 0:
        return result
    iou_mat = compute_iou_matrix(track_boxes, fight_boxes)
    # 对每个跟踪框，若与任一打架框 IoU >= 阈值，则标记为 True
    for idx, tid in enumerate(track_ids.tolist()):
        is_fight = False
        if iou_mat.shape[1] > 0:
            is_fight = bool((iou_mat[idx] >= iou_thr).any())
        result[int(tid)] = is_fight
    return result


class EventManager:
    def __init__(self, smooth_window: int, merge_gap: int):
        self.smooth_window = smooth_window
        self.merge_gap = merge_gap
        # 状态缓存：id -> 最近 smooth_window 帧的布尔队列
        self.buffers: Dict[int, List[int]] = {}
        # 事件区间：id -> List[Tuple[start, end]] (帧索引)
        self.events: Dict[int, List[Tuple[int, int]]] = {}
        # 当前进行中的片段：id -> start_frame or None
        self.active_start: Dict[int, Optional[int]] = {}

    def update(self, frame_idx: int, id_to_label: Dict[int, bool]):
        # 更新平滑缓冲与状态
        for tid, is_fight in id_to_label.items():
            buf = self.buffers.setdefault(tid, [])
            buf.append(1 if is_fight else 0)
            if len(buf) > self.smooth_window:
                buf.pop(0)
            # 平滑判断
            smoothed = sum(buf) > (len(buf) // 2)

            active = self.active_start.get(tid)
            if smoothed and active is None:
                # 新事件开始
                self.active_start[tid] = frame_idx
            elif (not smoothed) and active is not None:
                # 事件结束
                self._close_event(tid, active, frame_idx - 1)
                self.active_start[tid] = None

        # 对未出现在本帧的 ID，如果有进行中事件，缓冲按 0 推进
        present_ids = set(id_to_label.keys())
        for tid, active in list(self.active_start.items()):
            if active is not None and tid not in present_ids:
                buf = self.buffers.setdefault(tid, [])
                buf.append(0)
                if len(buf) > self.smooth_window:
                    buf.pop(0)
                smoothed = sum(buf) > (len(buf) // 2)
                if not smoothed:
                    self._close_event(tid, active, frame_idx - 1)
                    self.active_start[tid] = None

    def _close_event(self, tid: int, start_f: int, end_f: int):
        arr = self.events.setdefault(tid, [])
        # 合并与上一个片段的间隙
        if arr and start_f - arr[-1][1] <= self.merge_gap:
            prev_s, prev_e = arr[-1]
            arr[-1] = (prev_s, max(prev_e, end_f))
        else:
            arr.append((start_f, end_f))

    def finalize(self):
        # 关闭所有进行中事件
        for tid, active in list(self.active_start.items()):
            if active is not None:
                self._close_event(tid, active, active)  # 至少 1 帧
                self.active_start[tid] = None


def main():
    args = parse_args()

    # 加载模型
    det_model = YOLO(args.det_weights)
    fight_model = YOLO(args.fight_weights)

    # 跟踪：由 YOLO 内置 tracker 执行，另一模型做分类/检测
    tracker_cfg = resolve_tracker_path(args.tracker)

    # 事件管理器
    em = EventManager(smooth_window=args.smooth_window, merge_gap=args.merge_gap)

    # 使用 stream=True 获取逐帧结果
    track_stream = det_model.track(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf_det,
        iou=args.iou,
        device=args.device,
        tracker=tracker_cfg,
        persist=args.persist,
        vid_stride=args.vid_stride,
        stream=True,
        save=args.save_video,
        show=args.show,
        project=args.project,
        name=args.name,
    )

    # 叠加视频输出（可选）
    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / 'fight_overlay.mp4'
    writer: Optional[cv2.VideoWriter] = None

    # 自动检测 FPS
    fps_value = args.fps
    if fps_value <= 0 and isinstance(args.source, str):
        try:
            cap = cv2.VideoCapture(args.source)
            if cap.isOpened():
                fps_value = cap.get(cv2.CAP_PROP_FPS) or 0.0
            cap.release()
        except Exception:
            fps_value = 0.0
    if fps_value <= 0:
        fps_value = 25.0

    # 遍历每帧，使用 fight 模型推理并融合
    for frame_idx, det_res in enumerate(track_stream):
        # det_res.boxes.xyxy, det_res.boxes.id, det_res.orig_img
        boxes = getattr(det_res.boxes, 'xyxy', None)
        ids = getattr(det_res.boxes, 'id', None)
        frame = getattr(det_res, 'orig_img', None)
        if boxes is None or ids is None or frame is None:
            continue

        track_boxes = boxes.cpu().numpy() if hasattr(boxes, 'cpu') else np.asarray(boxes)
        track_ids = ids.cpu().numpy().astype(int) if hasattr(ids, 'cpu') else np.asarray(ids, dtype=int)

        # 计算 ID 标签
        id_to_label: Dict[int, bool]
        if args.fight_crop and track_boxes.size > 0:
            # 基于猪框裁剪 + 批量推理
            h, w = frame.shape[:2]
            crops: List[np.ndarray] = []
            id_list: List[int] = []
            for tid, (x1, y1, x2, y2) in zip(track_ids.tolist(), track_boxes):
                ix1 = int(max(0, min(w - 1, x1)))
                iy1 = int(max(0, min(h - 1, y1)))
                ix2 = int(max(0, min(w - 1, x2)))
                iy2 = int(max(0, min(h - 1, y2)))
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                crop = frame[iy1:iy2, ix1:ix2].copy()
                if crop.size == 0:
                    continue
                crops.append(crop)
                id_list.append(int(tid))

            id_to_label = {int(tid): False for tid in track_ids.tolist()}
            if crops:
                fight_results = fight_model.predict(
                    source=crops,
                    imgsz=args.imgsz,
                    conf=args.conf_fight,
                    iou=args.iou,
                    device=args.device,
                    verbose=False,
                    batch=args.fight_batch,
                )
                for idx, fr in enumerate(fight_results):
                    fboxes = getattr(fr.boxes, 'xyxy', None)
                    if fboxes is not None and len(fboxes) > 0:
                        id_to_label[id_list[idx]] = True
        else:
            # 整帧打架检测 + IoU 融合
            fight_results = fight_model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf_fight,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )
            fight_xyxy = []
            if fight_results:
                fr = fight_results[0]
                fboxes = getattr(fr.boxes, 'xyxy', None)
                if fboxes is not None:
                    fight_xyxy = (fboxes.cpu().numpy() if hasattr(fboxes, 'cpu') else np.asarray(fboxes))
            fight_xyxy = np.asarray(fight_xyxy, dtype=np.float32).reshape(-1, 4)
            id_to_label = fuse_fight_labels(track_ids, track_boxes, fight_xyxy, args.iou_match)

        # 事件更新
        em.update(frame_idx, id_to_label)

        # 叠加绘制（仅对打架目标画红框）
        if args.save_overlay:
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(overlay_path), fourcc, fps_value, (w, h))
            draw = frame.copy()
            for (x1, y1, x2, y2), tid in zip(track_boxes, track_ids):
                if id_to_label.get(int(tid), False):
                    cv2.rectangle(draw, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(draw, f'ID {int(tid)} FIGHT', (int(x1), int(max(0, y1) - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            writer.write(draw)

    # 收尾
    em.finalize()

    if writer is not None:
        writer.release()

    # 导出事件
    out_path = out_dir / 'fight_events.csv'
    with out_path.open('w', encoding='utf-8') as f:
        f.write('id,start_frame,end_frame\n')
        for tid, segs in em.events.items():
            for s, e in segs:
                f.write(f'{tid},{s},{e}\n')
    print(f'[完成] 事件文件已保存: {out_path.resolve()}')

    # 导出带时间戳的事件（秒）
    out_path_time = out_dir / 'fight_events_time.csv'
    with out_path_time.open('w', encoding='utf-8') as f:
        f.write('id,start_sec,end_sec\n')
        for tid, segs in em.events.items():
            for s, e in segs:
                f.write(f'{tid},{s / fps_value:.3f},{e / fps_value:.3f}\n')
    print(f'[完成] 事件时间文件已保存: {out_path_time.resolve()}')


if __name__ == '__main__':
    main()



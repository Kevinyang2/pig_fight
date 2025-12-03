from ultralytics import YOLO
import argparse
from pathlib import Path
import ultralytics as ul_pkg
import numpy as np
import json
from collections import defaultdict
from typing import List, Tuple, Dict
import cv2


class FightDetector:
    """基于跟踪结果的打架行为检测器"""
    
    def __init__(self, 
                 window_size: int = 30,  # 滑动窗口大小（帧数）
                 stride: int = 15,  # 滑动步长
                 distance_threshold: float = 100,  # 距离阈值（像素）
                 speed_threshold: float = 50,  # 速度阈值
                 min_fight_duration: int = 15):  # 最小打架持续帧数
        """
        Args:
            window_size: 滑动窗口大小（帧数）
            stride: 窗口滑动步长
            distance_threshold: 判断两只猪距离过近的阈值
            speed_threshold: 判断运动剧烈的速度阈值
            min_fight_duration: 最小打架持续帧数
        """
        self.window_size = window_size
        self.stride = stride
        self.distance_threshold = distance_threshold
        self.speed_threshold = speed_threshold
        self.min_fight_duration = min_fight_duration
        
        self.frame_data = []  # 存储每帧的跟踪数据
        
    def add_frame(self, frame_idx: int, boxes, track_ids):
        """添加一帧的跟踪结果
        
        Args:
            frame_idx: 帧索引
            boxes: 检测框 (N, 4) - [x1, y1, x2, y2]
            track_ids: 跟踪ID (N,)
        """
        frame_info = {
            'frame_idx': frame_idx,
            'objects': []
        }
        
        if boxes is not None and len(boxes) > 0:
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box[:4]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                
                frame_info['objects'].append({
                    'track_id': int(track_id),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center': [float(cx), float(cy)],
                    'size': [float(w), float(h)]
                })
        
        self.frame_data.append(frame_info)
    
    def is_fighting_in_window(self, start_idx: int, end_idx: int) -> Tuple[bool, float]:
        """判断窗口内是否存在打架行为
        
        Args:
            start_idx: 窗口起始帧索引
            end_idx: 窗口结束帧索引
            
        Returns:
            (is_fighting, confidence): 是否打架及置信度
        """
        if end_idx > len(self.frame_data):
            end_idx = len(self.frame_data)
        
        window_frames = self.frame_data[start_idx:end_idx]
        
        if len(window_frames) < 2:
            return False, 0.0
        
        # 特征1: 检查是否有两只或以上的猪距离过近
        close_contact_count = 0
        total_pairs = 0
        
        # 特征2: 检查运动速度变化
        track_speeds = defaultdict(list)
        
        for i, frame in enumerate(window_frames):
            objects = frame['objects']
            
            # 计算两两之间的距离
            if len(objects) >= 2:
                for j in range(len(objects)):
                    for k in range(j + 1, len(objects)):
                        obj1 = objects[j]
                        obj2 = objects[k]
                        
                        # 计算中心点距离
                        dist = np.sqrt(
                            (obj1['center'][0] - obj2['center'][0])**2 +
                            (obj1['center'][1] - obj2['center'][1])**2
                        )
                        
                        total_pairs += 1
                        if dist < self.distance_threshold:
                            close_contact_count += 1
            
            # 计算每个目标的速度
            if i > 0:
                prev_frame = window_frames[i - 1]
                for obj in objects:
                    track_id = obj['track_id']
                    # 在上一帧中找到相同ID
                    for prev_obj in prev_frame['objects']:
                        if prev_obj['track_id'] == track_id:
                            speed = np.sqrt(
                                (obj['center'][0] - prev_obj['center'][0])**2 +
                                (obj['center'][1] - prev_obj['center'][1])**2
                            )
                            track_speeds[track_id].append(speed)
                            break
        
        # 计算特征得分
        scores = []
        
        # 得分1: 距离过近的比例
        if total_pairs > 0:
            close_ratio = close_contact_count / total_pairs
            scores.append(min(close_ratio * 2, 1.0))  # 归一化到[0,1]
        
        # 得分2: 速度变化剧烈程度
        if track_speeds:
            avg_speeds = [np.mean(speeds) for speeds in track_speeds.values() if len(speeds) > 0]
            if avg_speeds:
                max_speed = max(avg_speeds)
                speed_score = min(max_speed / self.speed_threshold, 1.0)
                scores.append(speed_score)
        
        # 综合判断
        if scores:
            confidence = np.mean(scores)
            is_fighting = confidence > 0.5  # 阈值可调
            return is_fighting, confidence
        
        return False, 0.0
    
    def detect_fight_segments(self) -> List[Tuple[int, int, float]]:
        """使用滑动窗口检测所有打架片段
        
        Returns:
            List of (start_frame, end_frame, confidence)
        """
        fight_segments = []
        
        for start_idx in range(0, len(self.frame_data) - self.window_size + 1, self.stride):
            end_idx = start_idx + self.window_size
            is_fighting, confidence = self.is_fighting_in_window(start_idx, end_idx)
            
            if is_fighting:
                start_frame = self.frame_data[start_idx]['frame_idx']
                end_frame = self.frame_data[min(end_idx - 1, len(self.frame_data) - 1)]['frame_idx']
                fight_segments.append((start_frame, end_frame, confidence))
        
        # 合并重叠的片段
        if fight_segments:
            fight_segments = self._merge_segments(fight_segments)
        
        return fight_segments
    
    def _merge_segments(self, segments: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """合并重叠的打架片段"""
        if not segments:
            return []
        
        # 按开始时间排序
        segments = sorted(segments, key=lambda x: x[0])
        merged = [segments[0]]
        
        for current in segments[1:]:
            last = merged[-1]
            # 如果重叠，合并
            if current[0] <= last[1] + self.stride:
                merged[-1] = (
                    last[0],
                    max(last[1], current[1]),
                    max(last[2], current[2])  # 取较高的置信度
                )
            else:
                merged.append(current)
        
        # 过滤太短的片段
        merged = [seg for seg in merged if seg[1] - seg[0] >= self.min_fight_duration]
        
        return merged


class FightEvaluator:
    """打架检测评估器"""
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Args:
            iou_threshold: IoU阈值，用于判断检测片段是否匹配GT
        """
        self.iou_threshold = iou_threshold
    
    @staticmethod
    def load_ground_truth(gt_file: str, default_fps: float = 30.0) -> Dict[str, List[Tuple[int, int]]]:
        """加载ground truth
        
        支持多种GT文件格式：
        
        格式1（简单）:
        {
            "video1.mp4": [[start_frame1, end_frame1], [start_frame2, end_frame2], ...],
            "video2.mp4": [[start_frame1, end_frame1], ...]
        }
        
        格式2（带fps）:
        {
            "fps": 30,
            "video1.mp4": [[start_time1, end_time1], ...],  # 秒为单位
        }
        
        格式3（database格式，您的格式）:
        {
            "database": {
                "MyVideo_191": {
                    "subset": "test",
                    "annotations": [
                        {"segment": ["93.1", "98.2"], "label": "fight"},
                        ...
                    ]
                }
            }
        }
        
        Args:
            gt_file: GT文件路径
            default_fps: 默认帧率（用于格式3）
        """
        with open(gt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        gt_dict = {}
        
        # 检测格式3：database格式
        if 'database' in data:
            database = data['database']
            fps = data.get('fps', default_fps)  # 从文件读取fps或使用默认值
            
            for video_name, video_data in database.items():
                if 'annotations' in video_data:
                    segments = []
                    for ann in video_data['annotations']:
                        if 'segment' in ann and ann.get('label') == 'fight':
                            start, end = ann['segment']
                            # segment中的时间可能是字符串，转为浮点数（秒）
                            start_sec = float(start)
                            end_sec = float(end)
                            segments.append((start_sec, end_sec))
                    if segments:
                        gt_dict[video_name] = [(int(s * fps), int(e * fps)) for s, e in segments]
            return gt_dict
        
        # 格式1和2：简单格式
        fps = data.pop('fps', None)
        
        for video_name, segments in data.items():
            if fps is not None:
                # 转换时间到帧
                gt_dict[video_name] = [(int(s * fps), int(e * fps)) for s, e in segments]
            else:
                gt_dict[video_name] = [(int(s), int(e)) for s, e in segments]
        
        return gt_dict
    
    @staticmethod
    def calculate_iou(seg1: Tuple[int, int], seg2: Tuple[int, int]) -> float:
        """计算两个时间片段的IoU"""
        start1, end1 = seg1
        start2, end2 = seg2
        
        # 计算交集
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        intersection = max(0, intersection_end - intersection_start)
        
        # 计算并集
        union_start = min(start1, start2)
        union_end = max(end1, end2)
        union = union_end - union_start
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def evaluate(self, 
                 pred_segments: List[Tuple[int, int]], 
                 gt_segments: List[Tuple[int, int]]) -> Dict:
        """评估检测结果
        
        Args:
            pred_segments: 预测的打架片段 [(start, end), ...]
            gt_segments: Ground truth片段 [(start, end), ...]
            
        Returns:
            评估指标字典
        """
        if not gt_segments:
            return {
                'precision': 0.0 if pred_segments else 1.0,
                'recall': 0.0,
                'f1': 0.0,
                'tp': 0,
                'fp': len(pred_segments),
                'fn': 0,
                'iou_threshold': self.iou_threshold
            }
        
        if not pred_segments:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'tp': 0,
                'fp': 0,
                'fn': len(gt_segments),
                'iou_threshold': self.iou_threshold
            }
        
        # 匹配预测和GT
        matched_gt = set()
        matched_pred = set()
        
        for i, pred_seg in enumerate(pred_segments):
            for j, gt_seg in enumerate(gt_segments):
                if j in matched_gt:
                    continue
                
                iou = self.calculate_iou(pred_seg[:2], gt_seg)
                if iou >= self.iou_threshold:
                    matched_gt.add(j)
                    matched_pred.add(i)
                    break
        
        tp = len(matched_pred)
        fp = len(pred_segments) - tp
        fn = len(gt_segments) - len(matched_gt)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'iou_threshold': self.iou_threshold
        }


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO跟踪 + 打架检测与评估')
    
    # 原有跟踪参数
    parser.add_argument('--weights', type=str, 
                       default='runs/train/v10-APConv-AssemFormer-HSFPN-ATFLm_exp/weights/best.pt', 
                       help='模型权重')
    parser.add_argument('--source', type=str, required=True, 
                       help='视频/图像/目录/通配符')
    parser.add_argument('--tracker', type=str, default='botsort.yaml', 
                       help='跟踪器配置')
    parser.add_argument('--device', type=str, default='0', 
                       help='CUDA设备')
    parser.add_argument('--conf', type=float, default=0.5, 
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.6, 
                       help='IOU阈值')
    parser.add_argument('--imgsz', type=int, default=640, 
                       help='推理尺寸')
    parser.add_argument('--show', action='store_true', 
                       help='窗口显示')
    parser.add_argument('--save', action='store_true', 
                       help='保存可视化结果')
    parser.add_argument('--vid-stride', type=int, default=1, 
                       help='视频抽帧步长')
    parser.add_argument('--persist', action='store_true', 
                       help='保持跟踪ID')
    
    # 打架检测参数
    parser.add_argument('--window-size', type=int, default=30, 
                       help='滑动窗口大小（帧数）')
    parser.add_argument('--stride', type=int, default=15, 
                       help='窗口滑动步长')
    parser.add_argument('--distance-threshold', type=float, default=100, 
                       help='距离阈值（像素）')
    parser.add_argument('--speed-threshold', type=float, default=50, 
                       help='速度阈值')
    parser.add_argument('--min-fight-duration', type=int, default=15, 
                       help='最小打架持续帧数')
    
    # 评估参数
    parser.add_argument('--gt-file', type=str, default=None, 
                       help='Ground truth JSON文件路径')
    parser.add_argument('--eval-iou-threshold', type=float, default=0.5, 
                       help='评估时的IoU阈值')
    parser.add_argument('--output-dir', type=str, default='fight_detection_results', 
                       help='结果输出目录')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='视频帧率（用于GT时间转换，默认30）')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print(f"加载模型: {args.weights}")
    model = YOLO(args.weights)
    
    # 解析跟踪器路径
    def resolve_tracker_path(tracker: str) -> str:
        p = Path(tracker)
        if p.exists():
            return str(p)
        name = tracker.lower()
        if name.endswith('.yaml'):
            name = name[:-5]
        if name in ('bytetrack', 'botsort'):
            base = Path(ul_pkg.__file__).parent / 'cfg' / 'trackers'
            cand = base / f'{name}.yaml'
            if cand.exists():
                return str(cand)
        return tracker
    
    tracker_path = resolve_tracker_path(args.tracker)
    
    # 初始化打架检测器
    detector = FightDetector(
        window_size=args.window_size,
        stride=args.stride,
        distance_threshold=args.distance_threshold,
        speed_threshold=args.speed_threshold,
        min_fight_duration=args.min_fight_duration
    )
    
    # 运行跟踪
    print(f"开始跟踪: {args.source}")
    results = model.track(
        source=args.source,
        tracker=tracker_path,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        show=args.show,
        save=args.save,
        vid_stride=args.vid_stride,
        persist=args.persist,
        stream=True  # 流式处理
    )
    
    # 处理每一帧
    frame_idx = 0
    for result in results:
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy()
            detector.add_frame(frame_idx, boxes, track_ids)
        else:
            detector.add_frame(frame_idx, None, None)
        
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"已处理 {frame_idx} 帧")
    
    print(f"跟踪完成，共处理 {frame_idx} 帧")
    
    # 检测打架片段
    print("开始检测打架片段...")
    fight_segments = detector.detect_fight_segments()
    
    print(f"\n检测到 {len(fight_segments)} 个打架片段:")
    for i, (start, end, conf) in enumerate(fight_segments):
        print(f"  片段 {i+1}: 帧 {start}-{end} (置信度: {conf:.3f})")
    
    # 保存检测结果
    video_name = Path(args.source).name
    result_file = output_dir / f"{Path(video_name).stem}_predictions.json"
    
    pred_data = {
        'video': video_name,
        'segments': [[int(s), int(e), float(c)] for s, e, c in fight_segments],
        'total_frames': frame_idx,
        'parameters': {
            'window_size': args.window_size,
            'stride': args.stride,
            'distance_threshold': args.distance_threshold,
            'speed_threshold': args.speed_threshold,
            'min_fight_duration': args.min_fight_duration
        }
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(pred_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n预测结果已保存到: {result_file}")
    
    # 如果提供了GT，进行评估
    if args.gt_file:
        print(f"\n加载Ground Truth: {args.gt_file}")
        evaluator = FightEvaluator(iou_threshold=args.eval_iou_threshold)
        gt_dict = evaluator.load_ground_truth(args.gt_file, args.fps)
        
        # 使用不带扩展名的视频名查找GT（GT中的键名通常不包含扩展名）
        video_name_stem = Path(args.source).stem
        
        if video_name_stem in gt_dict:
            gt_segments = gt_dict[video_name_stem]
            print(f"GT包含 {len(gt_segments)} 个打架片段")
            
            # 评估
            pred_segments_for_eval = [(s, e) for s, e, c in fight_segments]
            metrics = evaluator.evaluate(pred_segments_for_eval, gt_segments)
            
            print("\n" + "="*50)
            print("评估结果:")
            print("="*50)
            print(f"精确率 (Precision): {metrics['precision']:.4f}")
            print(f"召回率 (Recall):    {metrics['recall']:.4f}")
            print(f"F1分数 (F1-Score):  {metrics['f1']:.4f}")
            print(f"真正例 (TP):        {metrics['tp']}")
            print(f"假正例 (FP):        {metrics['fp']}")
            print(f"假负例 (FN):        {metrics['fn']}")
            print(f"IoU阈值:            {metrics['iou_threshold']}")
            print("="*50)
            
            # 保存评估结果
            eval_file = output_dir / f"{Path(video_name).stem}_evaluation.json"
            eval_data = {
                'video': video_name,
                'metrics': metrics,
                'predictions': [[int(s), int(e), float(c)] for s, e, c in fight_segments],
                'ground_truth': [[int(s), int(e)] for s, e in gt_segments]
            }
            
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(eval_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n评估结果已保存到: {eval_file}")
        else:
            print(f"警告: GT文件中未找到视频 '{video_name_stem}'")
            print(f"提示: GT中可用的视频名: {list(gt_dict.keys())[:5]}...")  # 显示前5个
    
    return fight_segments


if __name__ == '__main__':
    main()


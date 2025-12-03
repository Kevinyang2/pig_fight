"""批量评估多个视频的打架检测性能"""

from pathlib import Path
import json
import argparse
from track_with_fight_detection import YOLO, FightDetector, FightEvaluator
import ultralytics as ul_pkg
import numpy as np


def resolve_tracker_path(tracker: str) -> str:
    """解析跟踪器路径"""
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


def process_single_video(video_path: str, 
                         model: YOLO, 
                         detector: FightDetector,
                         tracker_path: str,
                         args) -> list:
    """处理单个视频并返回检测结果"""
    
    print(f"\n处理视频: {video_path}")
    detector.frame_data = []  # 重置检测器
    
    # 运行跟踪
    results = model.track(
        source=video_path,
        tracker=tracker_path,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        show=False,  # 批量处理时不显示
        save=args.save,
        vid_stride=args.vid_stride,
        persist=args.persist,
        stream=True
    )
    
    # 收集跟踪结果
    frame_idx = 0
    for result in results:
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy()
            detector.add_frame(frame_idx, boxes, track_ids)
        else:
            detector.add_frame(frame_idx, None, None)
        frame_idx += 1
    
    print(f"  共处理 {frame_idx} 帧")
    
    # 检测打架片段
    fight_segments = detector.detect_fight_segments()
    print(f"  检测到 {len(fight_segments)} 个打架片段")
    
    return fight_segments


def parse_args():
    parser = argparse.ArgumentParser(description='批量评估打架检测')
    
    parser.add_argument('--weights', type=str, required=True,
                       help='模型权重')
    parser.add_argument('--video-dir', type=str, required=True,
                       help='视频文件夹路径')
    parser.add_argument('--gt-file', type=str, required=True,
                       help='Ground truth JSON文件')
    parser.add_argument('--tracker', type=str, default='botsort.yaml',
                       help='跟踪器配置')
    parser.add_argument('--device', type=str, default='0',
                       help='CUDA设备')
    parser.add_argument('--conf', type=float, default=0.75,
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.6,
                       help='IOU阈值')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='推理尺寸')
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
    parser.add_argument('--eval-iou-threshold', type=float, default=0.5,
                       help='评估IoU阈值')
    parser.add_argument('--output-dir', type=str, default='batch_evaluation_results',
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
    
    # 解析跟踪器
    tracker_path = resolve_tracker_path(args.tracker)
    
    # 加载GT
    print(f"加载Ground Truth: {args.gt_file}")
    evaluator = FightEvaluator(iou_threshold=args.eval_iou_threshold)
    gt_dict = evaluator.load_ground_truth(args.gt_file, args.fps)
    
    # 获取视频列表
    video_dir = Path(args.video_dir)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f'*{ext}'))
    
    if not video_files:
        print(f"错误: 在 {video_dir} 中未找到视频文件")
        return
    
    print(f"\n找到 {len(video_files)} 个视频文件")
    
    # 初始化检测器
    detector = FightDetector(
        window_size=args.window_size,
        stride=args.stride,
        distance_threshold=args.distance_threshold,
        speed_threshold=args.speed_threshold,
        min_fight_duration=args.min_fight_duration
    )
    
    # 存储所有结果
    all_results = []
    overall_metrics = {
        'tp': 0,
        'fp': 0,
        'fn': 0
    }
    
    # 处理每个视频
    for video_file in video_files:
        video_name_full = video_file.name
        video_name = video_file.stem  # 不带扩展名，用于匹配GT
        
        # 检查GT中是否有这个视频
        if video_name not in gt_dict:
            print(f"\n跳过 {video_name_full}: GT中未找到（查找键名: {video_name}）")
            continue
        
        # 处理视频
        fight_segments = process_single_video(
            str(video_file), 
            model, 
            detector, 
            tracker_path, 
            args
        )
        
        # 评估
        gt_segments = gt_dict[video_name]
        pred_segments_for_eval = [(s, e) for s, e, c in fight_segments]
        metrics = evaluator.evaluate(pred_segments_for_eval, gt_segments)
        
        # 累计指标
        overall_metrics['tp'] += metrics['tp']
        overall_metrics['fp'] += metrics['fp']
        overall_metrics['fn'] += metrics['fn']
        
        # 保存单个视频结果
        video_result = {
            'video': video_name,
            'metrics': metrics,
            'predictions': [[int(s), int(e), float(c)] for s, e, c in fight_segments],
            'ground_truth': [[int(s), int(e)] for s, e in gt_segments]
        }
        all_results.append(video_result)
        
        print(f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
    
    # 计算总体指标
    tp = overall_metrics['tp']
    fp = overall_metrics['fp']
    fn = overall_metrics['fn']
    
    overall_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    overall_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
                 if (overall_precision + overall_recall) > 0 else 0.0
    
    overall_summary = {
        'total_videos': len(all_results),
        'overall_metrics': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        },
        'per_video_results': all_results,
        'parameters': {
            'window_size': args.window_size,
            'stride': args.stride,
            'distance_threshold': args.distance_threshold,
            'speed_threshold': args.speed_threshold,
            'min_fight_duration': args.min_fight_duration,
            'eval_iou_threshold': args.eval_iou_threshold
        }
    }
    
    # 保存总体结果
    summary_file = output_dir / 'overall_evaluation.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)
    
    # 打印总结
    print("\n" + "="*60)
    print("批量评估总结")
    print("="*60)
    print(f"处理视频数: {len(all_results)}")
    print(f"总体精确率 (Precision): {overall_precision:.4f}")
    print(f"总体召回率 (Recall):    {overall_recall:.4f}")
    print(f"总体F1分数 (F1-Score):  {overall_f1:.4f}")
    print(f"总真正例 (TP):          {tp}")
    print(f"总假正例 (FP):          {fp}")
    print(f"总假负例 (FN):          {fn}")
    print("="*60)
    print(f"\n详细结果已保存到: {summary_file}")
    
    # 生成详细报告
    report_file = output_dir / 'evaluation_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("打架检测评估报告\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"模型权重: {args.weights}\n")
        f.write(f"处理视频数: {len(all_results)}\n\n")
        
        f.write("参数设置:\n")
        f.write(f"  - 窗口大小: {args.window_size} 帧\n")
        f.write(f"  - 滑动步长: {args.stride} 帧\n")
        f.write(f"  - 距离阈值: {args.distance_threshold} 像素\n")
        f.write(f"  - 速度阈值: {args.speed_threshold}\n")
        f.write(f"  - 最小打架时长: {args.min_fight_duration} 帧\n")
        f.write(f"  - 评估IoU阈值: {args.eval_iou_threshold}\n\n")
        
        f.write("总体性能指标:\n")
        f.write(f"  - 精确率 (Precision): {overall_precision:.4f}\n")
        f.write(f"  - 召回率 (Recall):    {overall_recall:.4f}\n")
        f.write(f"  - F1分数:             {overall_f1:.4f}\n")
        f.write(f"  - TP: {tp}, FP: {fp}, FN: {fn}\n\n")
        
        f.write("="*60 + "\n")
        f.write("各视频详细结果:\n")
        f.write("="*60 + "\n\n")
        
        for result in all_results:
            f.write(f"视频: {result['video']}\n")
            m = result['metrics']
            f.write(f"  Precision: {m['precision']:.4f}\n")
            f.write(f"  Recall:    {m['recall']:.4f}\n")
            f.write(f"  F1-Score:  {m['f1']:.4f}\n")
            f.write(f"  TP: {m['tp']}, FP: {m['fp']}, FN: {m['fn']}\n")
            f.write(f"  预测片段数: {len(result['predictions'])}\n")
            f.write(f"  GT片段数:   {len(result['ground_truth'])}\n")
            f.write("\n")
    
    print(f"详细报告已保存到: {report_file}")


if __name__ == '__main__':
    main()


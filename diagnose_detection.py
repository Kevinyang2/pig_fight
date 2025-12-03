"""诊断工具：分析为什么没有检测到打架"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import ultralytics as ul_pkg
import numpy as np
from collections import defaultdict


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


def diagnose(args):
    """诊断检测问题"""
    
    print("="*70)
    print("打架检测诊断工具")
    print("="*70)
    
    # 加载模型
    print(f"\n[1] 加载模型: {args.weights}")
    model = YOLO(args.weights)
    print("✓ 模型加载成功")
    
    # 解析跟踪器
    tracker_path = resolve_tracker_path(args.tracker)
    print(f"\n[2] 跟踪器: {tracker_path}")
    
    # 运行跟踪
    print(f"\n[3] 开始跟踪视频: {args.source}")
    print(f"    置信度阈值: {args.conf}")
    print(f"    设备: {args.device}")
    
    results = model.track(
        source=args.source,
        tracker=tracker_path,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        persist=True,
        stream=True,
        verbose=False
    )
    
    # 收集跟踪数据
    frame_data = []
    frame_idx = 0
    
    print("\n处理视频帧...")
    for result in results:
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy()
            
            frame_info = {
                'frame_idx': frame_idx,
                'num_objects': len(boxes),
                'track_ids': track_ids.tolist(),
                'boxes': boxes.tolist()
            }
            frame_data.append(frame_info)
        else:
            frame_data.append({
                'frame_idx': frame_idx,
                'num_objects': 0,
                'track_ids': [],
                'boxes': []
            })
        
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            print(f"  已处理 {frame_idx} 帧")
    
    print(f"\n✓ 跟踪完成，共处理 {frame_idx} 帧")
    
    # 分析跟踪结果
    print("\n" + "="*70)
    print("[4] 跟踪结果分析")
    print("="*70)
    
    # 统计每帧的目标数
    num_objects_per_frame = [f['num_objects'] for f in frame_data]
    frames_with_objects = sum(1 for n in num_objects_per_frame if n > 0)
    frames_with_multiple = sum(1 for n in num_objects_per_frame if n >= 2)
    
    print(f"\n目标检测统计:")
    print(f"  - 有检测到目标的帧数: {frames_with_objects}/{frame_idx} ({100*frames_with_objects/frame_idx:.1f}%)")
    print(f"  - 检测到2个或以上目标的帧数: {frames_with_multiple}/{frame_idx} ({100*frames_with_multiple/frame_idx:.1f}%)")
    
    if frames_with_objects == 0:
        print("\n⚠️  警告: 没有任何帧检测到目标！")
        print("   可能原因:")
        print("   1. 置信度阈值太高 (当前: {})".format(args.conf))
        print("   2. 模型不适合当前视频内容")
        print("   3. 视频质量问题")
        print("\n建议:")
        print("   - 尝试降低置信度: --conf 0.3")
        print("   - 使用 --show 参数查看检测效果")
        return
    
    if num_objects_per_frame:
        avg_objects = np.mean([n for n in num_objects_per_frame if n > 0])
        max_objects = max(num_objects_per_frame)
        print(f"  - 平均目标数（有目标的帧）: {avg_objects:.2f}")
        print(f"  - 最大目标数: {max_objects}")
    
    # 统计跟踪ID
    all_track_ids = set()
    for f in frame_data:
        all_track_ids.update(f['track_ids'])
    
    print(f"\n跟踪ID统计:")
    print(f"  - 总共跟踪到的不同ID数: {len(all_track_ids)}")
    
    if len(all_track_ids) == 0:
        print("\n⚠️  警告: 没有成功跟踪到任何目标ID！")
        print("   可能原因:")
        print("   1. 跟踪器配置问题")
        print("   2. 检测结果太少导致无法跟踪")
        return
    
    # 分析每个ID的持续时间
    id_frames = defaultdict(list)
    for f in frame_data:
        for tid in f['track_ids']:
            id_frames[tid].append(f['frame_idx'])
    
    id_durations = {tid: len(frames) for tid, frames in id_frames.items()}
    avg_duration = np.mean(list(id_durations.values()))
    
    print(f"  - ID平均持续帧数: {avg_duration:.1f}")
    print(f"  - 持续时间Top 5:")
    for tid, duration in sorted(id_durations.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      ID {int(tid)}: {duration} 帧")
    
    # 分析距离和速度
    print(f"\n" + "="*70)
    print("[5] 打架特征分析")
    print("="*70)
    
    distances = []
    speeds = []
    
    for i, frame in enumerate(frame_data):
        if frame['num_objects'] >= 2:
            boxes = frame['boxes']
            # 计算两两距离
            for j in range(len(boxes)):
                for k in range(j + 1, len(boxes)):
                    x1_c = (boxes[j][0] + boxes[j][2]) / 2
                    y1_c = (boxes[j][1] + boxes[j][3]) / 2
                    x2_c = (boxes[k][0] + boxes[k][2]) / 2
                    y2_c = (boxes[k][1] + boxes[k][3]) / 2
                    
                    dist = np.sqrt((x1_c - x2_c)**2 + (y1_c - y2_c)**2)
                    distances.append(dist)
        
        # 计算速度
        if i > 0 and frame['num_objects'] > 0:
            prev_frame = frame_data[i - 1]
            for tid in frame['track_ids']:
                if tid in prev_frame['track_ids']:
                    # 找到对应的box
                    curr_idx = frame['track_ids'].index(tid)
                    prev_idx = prev_frame['track_ids'].index(tid)
                    
                    curr_box = frame['boxes'][curr_idx]
                    prev_box = prev_frame['boxes'][prev_idx]
                    
                    curr_cx = (curr_box[0] + curr_box[2]) / 2
                    curr_cy = (curr_box[1] + curr_box[3]) / 2
                    prev_cx = (prev_box[0] + prev_box[2]) / 2
                    prev_cy = (prev_box[1] + prev_box[3]) / 2
                    
                    speed = np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)
                    speeds.append(speed)
    
    print(f"\n距离统计（目标间距离，像素）:")
    if distances:
        print(f"  - 最小距离: {min(distances):.1f}")
        print(f"  - 平均距离: {np.mean(distances):.1f}")
        print(f"  - 中位距离: {np.median(distances):.1f}")
        print(f"  - 最大距离: {max(distances):.1f}")
        print(f"  - 小于100像素的比例: {100*sum(1 for d in distances if d < 100)/len(distances):.1f}%")
        print(f"  - 小于150像素的比例: {100*sum(1 for d in distances if d < 150)/len(distances):.1f}%")
        
        print(f"\n  当前distance_threshold设置: {args.distance_threshold}")
        close_ratio = sum(1 for d in distances if d < args.distance_threshold) / len(distances)
        print(f"  → 符合当前阈值的距离比例: {100*close_ratio:.1f}%")
        
        if close_ratio < 0.1:
            print(f"\n  ⚠️  警告: 只有很少的距离小于阈值！")
            print(f"     建议降低distance_threshold到: {int(np.percentile(distances, 30))}")
    else:
        print("  ⚠️  没有计算到任何距离（可能同时出现的目标少于2个）")
    
    print(f"\n速度统计（帧间位移，像素/帧）:")
    if speeds:
        print(f"  - 最小速度: {min(speeds):.1f}")
        print(f"  - 平均速度: {np.mean(speeds):.1f}")
        print(f"  - 中位速度: {np.median(speeds):.1f}")
        print(f"  - 最大速度: {max(speeds):.1f}")
        print(f"  - 速度 > 50 的比例: {100*sum(1 for s in speeds if s > 50)/len(speeds):.1f}%")
        print(f"  - 速度 > 30 的比例: {100*sum(1 for s in speeds if s > 30)/len(speeds):.1f}%")
        
        print(f"\n  当前speed_threshold设置: {args.speed_threshold}")
        fast_ratio = sum(1 for s in speeds if s > args.speed_threshold) / len(speeds)
        print(f"  → 符合当前阈值的速度比例: {100*fast_ratio:.1f}%")
        
        if fast_ratio < 0.1:
            print(f"\n  ⚠️  警告: 只有很少的速度大于阈值！")
            print(f"     建议降低speed_threshold到: {int(np.percentile(speeds, 70))}")
    else:
        print("  ⚠️  没有计算到任何速度")
    
    # 给出建议
    print(f"\n" + "="*70)
    print("[6] 诊断建议")
    print("="*70)
    
    suggestions = []
    
    if frames_with_multiple < frame_idx * 0.3:
        suggestions.append("• 同时出现多个目标的帧较少，可能影响打架检测")
        suggestions.append("  → 检查模型是否能稳定检测到多只猪")
    
    if distances and close_ratio < 0.2:
        new_threshold = int(np.percentile(distances, 40))
        suggestions.append(f"• 目标间距离较大，当前distance_threshold可能太严格")
        suggestions.append(f"  → 建议: --distance-threshold {new_threshold}")
    
    if speeds and fast_ratio < 0.2:
        new_threshold = int(np.percentile(speeds, 70))
        suggestions.append(f"• 运动速度较慢，当前speed_threshold可能太严格")
        suggestions.append(f"  → 建议: --speed-threshold {new_threshold}")
    
    if len(suggestions) == 0:
        print("\n✓ 跟踪数据看起来正常")
        print("  如果仍然检测不到打架，可以尝试:")
        print("  1. 调整window_size: --window-size 40")
        print("  2. 减小stride: --stride 10")
        print("  3. 减小min_fight_duration: --min-fight-duration 10")
    else:
        print("\n发现以下问题:")
        for s in suggestions:
            print(s)
    
    # 生成推荐命令
    print(f"\n" + "="*70)
    print("[7] 推荐命令")
    print("="*70)
    
    if distances and speeds:
        rec_dist = int(np.percentile(distances, 40))
        rec_speed = int(np.percentile(speeds, 70))
        
        print(f"\n基于当前视频分析，建议使用以下参数:")
        print(f"""
python track_with_fight_detection.py \\
    --weights {args.weights} \\
    --source {args.source} \\
    --gt-file ground_truth_example.json \\
    --fps 30 \\
    --distance-threshold {rec_dist} \\
    --speed-threshold {rec_speed} \\
    --window-size 30 \\
    --stride 10 \\
    --min-fight-duration 10 \\
    --show
""")
    
    print("="*70)


def parse_args():
    parser = argparse.ArgumentParser(description='诊断打架检测问题')
    parser.add_argument('--weights', type=str, required=True, help='模型权重')
    parser.add_argument('--source', type=str, required=True, help='视频文件')
    parser.add_argument('--tracker', type=str, default='botsort.yaml', help='跟踪器配置')
    parser.add_argument('--device', type=str, default='0', help='设备')
    parser.add_argument('--conf', type=float, default=0.75, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.6, help='IOU阈值')
    parser.add_argument('--imgsz', type=int, default=640, help='推理尺寸')
    parser.add_argument('--distance-threshold', type=float, default=100, help='当前的距离阈值')
    parser.add_argument('--speed-threshold', type=float, default=50, help='当前的速度阈值')
    return parser.parse_args()


def main():
    args = parse_args()
    diagnose(args)


if __name__ == '__main__':
    main()


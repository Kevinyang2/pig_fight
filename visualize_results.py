"""可视化检测结果的工具"""

import json
import cv2
import argparse
from pathlib import Path
import numpy as np


def load_predictions(pred_file: str):
    """加载预测结果"""
    with open(pred_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_ground_truth(gt_file: str, video_name: str, default_fps: float = 30.0):
    """加载ground truth
    
    支持多种格式，包括database格式
    """
    with open(gt_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检测database格式
    if 'database' in data:
        database = data['database']
        fps = data.get('fps', default_fps)
        
        if video_name in database:
            video_data = database[video_name]
            if 'annotations' in video_data:
                segments = []
                for ann in video_data['annotations']:
                    if 'segment' in ann and ann.get('label') == 'fight':
                        start, end = ann['segment']
                        start_sec = float(start)
                        end_sec = float(end)
                        # 转换为帧号
                        segments.append((int(start_sec * fps), int(end_sec * fps)))
                return segments if segments else None
        return None
    
    # 简单格式
    fps = data.get('fps', None)
    if video_name in data and video_name != 'fps':
        segments = data[video_name]
        if fps is not None:
            # 时间转帧
            return [(int(s * fps), int(e * fps)) for s, e in segments]
        else:
            return [(int(s), int(e)) for s, e in segments]
    
    return None


def draw_timeline(frame, current_frame, total_frames, pred_segments, gt_segments=None, 
                 timeline_height=60, margin=20):
    """在帧上绘制时间轴"""
    h, w = frame.shape[:2]
    
    # 创建时间轴区域
    timeline_y = h - timeline_height - margin
    timeline_start_x = margin
    timeline_end_x = w - margin
    timeline_width = timeline_end_x - timeline_start_x
    
    # 绘制背景
    cv2.rectangle(frame, (timeline_start_x, timeline_y), 
                 (timeline_end_x, timeline_y + timeline_height), 
                 (40, 40, 40), -1)
    
    # 绘制GT片段（如果有）
    if gt_segments:
        for start, end in gt_segments:
            x1 = int(timeline_start_x + (start / total_frames) * timeline_width)
            x2 = int(timeline_start_x + (end / total_frames) * timeline_width)
            cv2.rectangle(frame, (x1, timeline_y + 5), (x2, timeline_y + 25), 
                         (0, 255, 0), -1)  # 绿色：GT
    
    # 绘制预测片段
    for seg in pred_segments:
        start, end = seg[0], seg[1]
        x1 = int(timeline_start_x + (start / total_frames) * timeline_width)
        x2 = int(timeline_start_x + (end / total_frames) * timeline_width)
        
        # 红色：预测
        y_offset = 30 if gt_segments else 5
        cv2.rectangle(frame, (x1, timeline_y + y_offset), (x2, timeline_y + y_offset + 20), 
                     (0, 0, 255), -1)
    
    # 绘制当前位置指示线
    current_x = int(timeline_start_x + (current_frame / total_frames) * timeline_width)
    cv2.line(frame, (current_x, timeline_y), (current_x, timeline_y + timeline_height), 
            (255, 255, 255), 2)
    
    # 添加图例
    legend_y = timeline_y - 30
    if gt_segments:
        cv2.rectangle(frame, (timeline_start_x, legend_y), 
                     (timeline_start_x + 20, legend_y + 15), (0, 255, 0), -1)
        cv2.putText(frame, "GT", (timeline_start_x + 25, legend_y + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.rectangle(frame, (timeline_start_x + 80, legend_y), 
                     (timeline_start_x + 100, legend_y + 15), (0, 0, 255), -1)
        cv2.putText(frame, "Pred", (timeline_start_x + 105, legend_y + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    else:
        cv2.rectangle(frame, (timeline_start_x, legend_y), 
                     (timeline_start_x + 20, legend_y + 15), (0, 0, 255), -1)
        cv2.putText(frame, "Fight", (timeline_start_x + 25, legend_y + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame


def is_in_segment(frame_idx, segments):
    """判断帧是否在某个片段内"""
    for seg in segments:
        start, end = seg[0], seg[1]
        if start <= frame_idx <= end:
            return True
    return False


def visualize_video(video_path: str, pred_file: str, gt_file: str = None, 
                   output_video: str = None, show: bool = True, fps: float = 30.0):
    """可视化视频中的检测结果"""
    
    # 加载预测结果
    pred_data = load_predictions(pred_file)
    pred_segments = pred_data.get('segments', [])
    
    print(f"加载预测结果: {len(pred_segments)} 个片段")
    
    # 加载GT（如果有）
    gt_segments = None
    if gt_file:
        video_name = Path(video_path).stem  # 不带扩展名，用于匹配GT
        gt_segments = load_ground_truth(gt_file, video_name, fps)
        if gt_segments:
            print(f"加载GT: {len(gt_segments)} 个片段")
        else:
            print(f"警告: GT中未找到视频 '{video_name}'")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息: {width}x{height}, {fps:.2f} FPS, {total_frames} 帧")
    
    # 准备输出视频
    writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        print(f"将保存到: {output_video}")
    
    frame_idx = 0
    paused = False
    
    print("\n控制:")
    print("  SPACE - 暂停/继续")
    print("  A/D - 后退/前进10帧（暂停时）")
    print("  Q - 退出")
    print()
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检查当前帧是否在打架片段内
            in_pred = is_in_segment(frame_idx, pred_segments)
            in_gt = is_in_segment(frame_idx, gt_segments) if gt_segments else False
            
            # 添加状态文本
            status_text = f"Frame: {frame_idx}/{total_frames} | Time: {frame_idx/fps:.2f}s"
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 添加检测状态
            if in_pred:
                cv2.putText(frame, "FIGHTING (Predicted)", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if in_gt:
                cv2.putText(frame, "FIGHTING (GT)", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 添加匹配状态
            if in_pred and in_gt:
                cv2.putText(frame, "TRUE POSITIVE", (width - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif in_pred and not in_gt and gt_segments:
                cv2.putText(frame, "FALSE POSITIVE", (width - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            elif not in_pred and in_gt:
                cv2.putText(frame, "FALSE NEGATIVE", (width - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # 绘制时间轴
            frame = draw_timeline(frame, frame_idx, total_frames, 
                                 pred_segments, gt_segments)
            
            if writer:
                writer.write(frame)
            
            if show:
                # 缩放以适应屏幕
                h, w = frame.shape[:2]
                max_h, max_w = 720, 1280
                if h > max_h or w > max_w:
                    scale = min(max_w/w, max_h/h)
                    display_frame = cv2.resize(frame, None, fx=scale, fy=scale)
                else:
                    display_frame = frame
                
                cv2.imshow('Fight Detection Visualization', display_frame)
            
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"进度: {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.1f}%)")
        
        else:
            # 暂停时重新显示当前帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                in_pred = is_in_segment(frame_idx, pred_segments)
                in_gt = is_in_segment(frame_idx, gt_segments) if gt_segments else False
                
                status_text = f"Frame: {frame_idx}/{total_frames} | Time: {frame_idx/fps:.2f}s [PAUSED]"
                cv2.putText(frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                if in_pred:
                    cv2.putText(frame, "FIGHTING (Predicted)", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if in_gt:
                    cv2.putText(frame, "FIGHTING (GT)", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                frame = draw_timeline(frame, frame_idx, total_frames, 
                                     pred_segments, gt_segments)
                
                h, w = frame.shape[:2]
                max_h, max_w = 720, 1280
                if h > max_h or w > max_w:
                    scale = min(max_w/w, max_h/h)
                    display_frame = cv2.resize(frame, None, fx=scale, fy=scale)
                else:
                    display_frame = frame
                
                cv2.imshow('Fight Detection Visualization', display_frame)
        
        # 处理按键
        wait_time = 1 if not paused else 0
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord(' '):
            paused = not paused
            print(f"{'已暂停' if paused else '继续播放'} at frame {frame_idx}")
        elif key == ord('a') or key == ord('A'):
            if paused:
                frame_idx = max(0, frame_idx - 10)
                print(f"后退到帧 {frame_idx}")
        elif key == ord('d') or key == ord('D'):
            if paused:
                frame_idx = min(total_frames - 1, frame_idx + 10)
                print(f"前进到帧 {frame_idx}")
    
    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    
    print("\n可视化完成")


def parse_args():
    parser = argparse.ArgumentParser(description='可视化打架检测结果')
    parser.add_argument('--video', type=str, required=True,
                       help='视频文件路径')
    parser.add_argument('--pred', type=str, required=True,
                       help='预测结果JSON文件')
    parser.add_argument('--gt', type=str, default=None,
                       help='Ground Truth JSON文件（可选）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出视频路径（可选）')
    parser.add_argument('--no-show', action='store_true',
                       help='不显示窗口（仅保存）')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='视频帧率（用于GT时间转换，默认30）')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查文件是否存在
    if not Path(args.video).exists():
        print(f"错误: 视频文件不存在: {args.video}")
        return
    
    if not Path(args.pred).exists():
        print(f"错误: 预测文件不存在: {args.pred}")
        return
    
    if args.gt and not Path(args.gt).exists():
        print(f"警告: GT文件不存在: {args.gt}")
        args.gt = None
    
    visualize_video(
        args.video,
        args.pred,
        args.gt,
        args.output,
        show=not args.no_show,
        fps=args.fps
    )


if __name__ == '__main__':
    main()


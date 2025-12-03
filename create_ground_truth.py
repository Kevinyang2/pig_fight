"""交互式创建Ground Truth标注文件的工具"""

import cv2
import json
from pathlib import Path
import argparse


class GTAnnotator:
    """Ground Truth标注工具"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.video_name = Path(video_path).name
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        self.segments = []
        self.current_frame = 0
        
        print(f"\n视频信息:")
        print(f"  文件: {self.video_name}")
        print(f"  帧率: {self.fps:.2f} FPS")
        print(f"  总帧数: {self.total_frames}")
        print(f"  时长: {self.duration:.2f} 秒")
    
    def frame_to_time(self, frame: int) -> float:
        """帧号转时间"""
        return frame / self.fps if self.fps > 0 else 0
    
    def time_to_frame(self, time: float) -> int:
        """时间转帧号"""
        return int(time * self.fps)
    
    def show_frame(self, frame_idx: int):
        """显示指定帧"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            # 添加信息文本
            text = f"Frame: {frame_idx}/{self.total_frames} | Time: {self.frame_to_time(frame_idx):.2f}s"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            # 缩放以适应屏幕
            h, w = frame.shape[:2]
            max_h, max_w = 720, 1280
            if h > max_h or w > max_w:
                scale = min(max_w/w, max_h/h)
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            
            cv2.imshow('GT Annotator - Press Q to quit, SPACE to pause', frame)
            return True
        return False
    
    def annotate_interactive(self):
        """交互式标注"""
        print("\n交互式标注模式:")
        print("  - 播放视频，当看到打架开始时按 'S'（Start）")
        print("  - 打架结束时按 'E'（End）")
        print("  - 按 SPACE 暂停/继续")
        print("  - 按 'A'/'D' 前进/后退10帧（暂停时）")
        print("  - 按 'Q' 退出标注")
        print()
        
        playing = True
        marking_start = False
        temp_start = None
        
        while True:
            if playing:
                ret = self.show_frame(self.current_frame)
                if not ret:
                    break
                self.current_frame += 1
                if self.current_frame >= self.total_frames:
                    break
            else:
                self.show_frame(self.current_frame)
            
            key = cv2.waitKey(30 if playing else 0) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord(' '):  # 空格：暂停/继续
                playing = not playing
                print(f"{'继续播放' if playing else '已暂停'} at frame {self.current_frame}")
            elif key == ord('s') or key == ord('S'):  # 标记开始
                temp_start = self.current_frame
                marking_start = True
                print(f"打架开始标记: 帧 {temp_start} ({self.frame_to_time(temp_start):.2f}s)")
            elif key == ord('e') or key == ord('E'):  # 标记结束
                if marking_start and temp_start is not None:
                    temp_end = self.current_frame
                    self.segments.append([temp_start, temp_end])
                    print(f"打架结束标记: 帧 {temp_end} ({self.frame_to_time(temp_end):.2f}s)")
                    print(f"  -> 添加片段: [{temp_start}, {temp_end}]")
                    marking_start = False
                    temp_start = None
                else:
                    print("错误: 请先按 'S' 标记开始")
            elif key == ord('a') or key == ord('A'):  # 后退
                if not playing:
                    self.current_frame = max(0, self.current_frame - 10)
                    print(f"后退到帧 {self.current_frame}")
            elif key == ord('d') or key == ord('D'):  # 前进
                if not playing:
                    self.current_frame = min(self.total_frames - 1, self.current_frame + 10)
                    print(f"前进到帧 {self.current_frame}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        return self.segments
    
    def annotate_manual(self):
        """手动输入标注"""
        print("\n手动输入模式:")
        print("  输入打架片段的开始和结束时间")
        print("  格式: start_time end_time (秒) 或 start_frame end_frame (帧)")
        print("  输入 'done' 完成标注\n")
        
        use_frames = input("使用帧号还是时间? (f=帧号, t=时间) [f]: ").lower() != 't'
        
        while True:
            user_input = input(f"\n输入片段 ({'帧号' if use_frames else '时间'}，格式: start end): ").strip()
            
            if user_input.lower() == 'done':
                break
            
            try:
                parts = user_input.split()
                if len(parts) != 2:
                    print("错误: 请输入两个数字（开始 结束）")
                    continue
                
                start, end = map(float, parts)
                
                if use_frames:
                    start_frame, end_frame = int(start), int(end)
                else:
                    start_frame = self.time_to_frame(start)
                    end_frame = self.time_to_frame(end)
                
                if start_frame >= end_frame:
                    print("错误: 结束时间必须大于开始时间")
                    continue
                
                if start_frame < 0 or end_frame > self.total_frames:
                    print(f"错误: 帧号超出范围 (0-{self.total_frames})")
                    continue
                
                self.segments.append([start_frame, end_frame])
                print(f"  -> 添加片段: 帧 [{start_frame}, {end_frame}] "
                      f"({self.frame_to_time(start_frame):.2f}s - {self.frame_to_time(end_frame):.2f}s)")
                
            except ValueError:
                print("错误: 输入无效，请输入数字")
        
        return self.segments


def parse_args():
    parser = argparse.ArgumentParser(description='创建Ground Truth标注文件')
    parser.add_argument('--video', type=str, help='单个视频文件路径')
    parser.add_argument('--video-dir', type=str, help='视频文件夹路径（批量标注）')
    parser.add_argument('--output', type=str, default='ground_truth.json',
                       help='输出JSON文件路径')
    parser.add_argument('--mode', type=str, choices=['interactive', 'manual'], 
                       default='interactive',
                       help='标注模式：interactive=交互式播放, manual=手动输入')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 收集视频文件
    video_files = []
    if args.video:
        video_files.append(Path(args.video))
    elif args.video_dir:
        video_dir = Path(args.video_dir)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        for ext in video_extensions:
            video_files.extend(video_dir.glob(f'*{ext}'))
    else:
        print("错误: 请指定 --video 或 --video-dir")
        return
    
    if not video_files:
        print("错误: 未找到视频文件")
        return
    
    print(f"\n找到 {len(video_files)} 个视频文件")
    
    # 加载现有GT（如果存在）
    output_file = Path(args.output)
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        print(f"\n加载现有GT文件: {output_file}")
    else:
        gt_data = {}
    
    # 标注每个视频
    for video_file in video_files:
        video_name = video_file.name
        
        print("\n" + "="*60)
        print(f"标注视频: {video_name}")
        print("="*60)
        
        # 检查是否已标注
        if video_name in gt_data:
            print(f"该视频已标注，包含 {len(gt_data[video_name])} 个片段")
            choice = input("是否重新标注? (y/n) [n]: ").lower()
            if choice != 'y':
                continue
        
        annotator = GTAnnotator(str(video_file))
        
        if args.mode == 'interactive':
            segments = annotator.annotate_interactive()
        else:
            segments = annotator.annotate_manual()
        
        if segments:
            gt_data[video_name] = segments
            print(f"\n{video_name} 标注完成，共 {len(segments)} 个片段")
        else:
            print(f"\n{video_name} 未添加任何片段")
    
    # 保存GT文件
    # 询问是否需要添加fps信息
    add_fps = input("\n是否添加fps信息到GT文件? (y/n) [n]: ").lower() == 'y'
    if add_fps:
        fps = float(input("输入帧率 (FPS): "))
        gt_data['fps'] = fps
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(gt_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nGround Truth已保存到: {output_file}")
    print(f"共标注 {len([k for k in gt_data.keys() if k != 'fps'])} 个视频")
    
    # 显示统计
    total_segments = sum(len(v) for k, v in gt_data.items() if k != 'fps')
    print(f"总片段数: {total_segments}")


if __name__ == '__main__':
    main()


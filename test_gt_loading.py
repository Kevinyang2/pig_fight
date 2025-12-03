"""测试Ground Truth文件加载
用于验证GT文件格式是否正确，以及与视频文件的匹配情况
"""

import json
import argparse
from pathlib import Path
import cv2


def load_and_verify_gt(gt_file: str, video_dir: str = None, fps: float = 30.0):
    """加载并验证GT文件"""
    
    print("="*60)
    print("Ground Truth 文件验证工具")
    print("="*60)
    
    # 1. 加载GT文件
    print(f"\n[步骤1] 加载GT文件: {gt_file}")
    try:
        with open(gt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("✓ GT文件加载成功")
    except FileNotFoundError:
        print(f"✗ 错误: 文件不存在 - {gt_file}")
        return
    except json.JSONDecodeError as e:
        print(f"✗ 错误: JSON格式错误 - {e}")
        return
    
    # 2. 检测GT格式
    print(f"\n[步骤2] 检测GT格式")
    
    if 'database' in data:
        print("✓ 检测到database格式")
        database = data['database']
        gt_fps = data.get('fps', fps)
        print(f"  - FPS: {gt_fps} (从{'文件' if 'fps' in data else '参数'}获取)")
        
        video_list = list(database.keys())
        print(f"  - 视频数量: {len(video_list)}")
        
        # 统计打架片段
        total_segments = 0
        total_duration = 0
        
        print(f"\n[步骤3] GT内容概览")
        print(f"{'视频名称':<20} {'片段数':>8} {'总时长(秒)':>12}")
        print("-"*60)
        
        for video_name in sorted(video_list):
            video_data = database[video_name]
            if 'annotations' in video_data:
                segments = [ann for ann in video_data['annotations'] 
                           if ann.get('label') == 'fight']
                duration = 0
                for seg in segments:
                    start, end = float(seg['segment'][0]), float(seg['segment'][1])
                    duration += (end - start)
                
                print(f"{video_name:<20} {len(segments):>8} {duration:>12.1f}")
                total_segments += len(segments)
                total_duration += duration
        
        print("-"*60)
        print(f"{'总计':<20} {total_segments:>8} {total_duration:>12.1f}")
        print(f"\n统计信息:")
        print(f"  - 总视频数: {len(video_list)}")
        print(f"  - 总片段数: {total_segments}")
        print(f"  - 总时长: {total_duration:.1f}秒 ({total_duration/60:.1f}分钟)")
        print(f"  - 平均每视频: {total_segments/len(video_list):.1f}个片段")
        if total_segments > 0:
            print(f"  - 平均片段时长: {total_duration/total_segments:.1f}秒")
        
        # 3. 如果提供了视频目录，验证匹配
        if video_dir:
            print(f"\n[步骤4] 验证与视频文件的匹配")
            video_path = Path(video_dir)
            
            if not video_path.exists():
                print(f"✗ 错误: 视频目录不存在 - {video_dir}")
                return
            
            # 获取视频文件
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            video_files = []
            for ext in video_extensions:
                video_files.extend(video_path.glob(f'*{ext}'))
            
            video_files_dict = {f.stem: f for f in video_files}
            
            print(f"\n找到 {len(video_files)} 个视频文件")
            
            matched = 0
            missing = []
            extra = []
            
            # 检查GT中的视频是否有对应文件
            for video_name in video_list:
                if video_name in video_files_dict:
                    matched += 1
                    # 验证视频信息
                    video_file = video_files_dict[video_name]
                    cap = cv2.VideoCapture(str(video_file))
                    
                    if cap.isOpened():
                        video_fps = cap.get(cv2.CAP_PROP_FPS)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = total_frames / video_fps if video_fps > 0 else 0
                        cap.release()
                        
                        # 检查GT片段是否在视频范围内
                        video_data = database[video_name]
                        out_of_range = []
                        
                        if 'annotations' in video_data:
                            for ann in video_data['annotations']:
                                if ann.get('label') == 'fight':
                                    start, end = float(ann['segment'][0]), float(ann['segment'][1])
                                    if end > duration:
                                        out_of_range.append((start, end))
                        
                        status = "✓" if not out_of_range else "⚠"
                        fps_match = "✓" if abs(video_fps - gt_fps) < 0.1 else f"⚠ ({video_fps:.1f} fps)"
                        
                        print(f"  {status} {video_name:<20} {duration:>8.1f}s  FPS: {fps_match}")
                        
                        if out_of_range:
                            for start, end in out_of_range:
                                print(f"      ⚠ 标注超出范围: [{start:.1f}, {end:.1f}]秒 > {duration:.1f}秒")
                    else:
                        print(f"  ✗ {video_name:<20} (无法打开)")
                else:
                    missing.append(video_name)
            
            # 检查是否有多余的视频文件
            for video_stem in video_files_dict.keys():
                if video_stem not in video_list:
                    extra.append(video_stem)
            
            print(f"\n匹配结果:")
            print(f"  - 匹配成功: {matched}/{len(video_list)}")
            
            if missing:
                print(f"\n  GT中有但视频文件缺失 ({len(missing)}个):")
                for name in missing[:10]:  # 最多显示10个
                    print(f"    - {name}")
                if len(missing) > 10:
                    print(f"    ... 还有{len(missing)-10}个")
            
            if extra:
                print(f"\n  视频文件存在但GT中没有 ({len(extra)}个):")
                for name in extra[:10]:
                    print(f"    - {name}")
                if len(extra) > 10:
                    print(f"    ... 还有{len(extra)-10}个")
            
            if matched == len(video_list) and not extra:
                print("\n✓ 完美匹配！所有GT视频都有对应的视频文件")
        
    elif 'fps' in data:
        print("✓ 检测到简单时间格式（带fps）")
        fps = data['fps']
        video_list = [k for k in data.keys() if k != 'fps']
        print(f"  - FPS: {fps}")
        print(f"  - 视频数量: {len(video_list)}")
        
        print(f"\n视频列表:")
        for video_name in video_list:
            segments = data[video_name]
            print(f"  - {video_name}: {len(segments)}个片段")
    
    else:
        print("✓ 检测到简单帧号格式")
        video_list = list(data.keys())
        print(f"  - 视频数量: {len(video_list)}")
        
        print(f"\n视频列表:")
        for video_name in video_list:
            segments = data[video_name]
            print(f"  - {video_name}: {len(segments)}个片段")
    
    print("\n" + "="*60)
    print("验证完成")
    print("="*60)


def parse_args():
    parser = argparse.ArgumentParser(description='测试GT文件加载')
    parser.add_argument('--gt-file', type=str, required=True,
                       help='Ground Truth JSON文件路径')
    parser.add_argument('--video-dir', type=str, default=None,
                       help='视频文件夹路径（可选，用于验证匹配）')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='默认帧率（当GT文件中没有fps时使用）')
    return parser.parse_args()


def main():
    args = parse_args()
    load_and_verify_gt(args.gt_file, args.video_dir, args.fps)


if __name__ == '__main__':
    main()


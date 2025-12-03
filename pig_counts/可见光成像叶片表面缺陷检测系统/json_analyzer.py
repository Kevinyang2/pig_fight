"""
JSON数据分析模块
将检测结果JSON转换为用户友好的可读报告
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class DetectionAnalyzer:
    """检测结果分析器"""
    
    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def analyze_batch_results(self) -> str:
        """分析批量检测结果"""
        if "batch_info" not in self.data:
            return self._analyze_single_image()
        
        batch_info = self.data["batch_info"]
        results = self.data.get("results", [])
        
        # 统计数据
        total_images = batch_info.get("total_images", len(results))
        success_count = batch_info.get("success", 0)
        failed_count = batch_info.get("failed", 0)
        confidence_threshold = batch_info.get("confidence_threshold", 0.5)
        timestamp = batch_info.get("timestamp", "未知")
        
        # 统计所有类别
        class_stats = {}
        total_detections = 0
        
        for result in results:
            detections = result.get("detections", {})
            class_counts = detections.get("class_counts", {})
            
            for class_name, count in class_counts.items():
                if class_name not in class_stats:
                    class_stats[class_name] = {"count": 0, "images": 0}
                class_stats[class_name]["count"] += count
                class_stats[class_name]["images"] += 1
                total_detections += count
        
        # 生成报告
        report = []
        report.append("=" * 60)
        report.append("      太赫兹成像内部探伤检测 - 批量分析报告")
        report.append("=" * 60)
        report.append("")
        report.append(f"检测时间：{timestamp}")
        report.append(f"置信度阈值：{confidence_threshold:.2f}")
        report.append("")
        
        report.append("━" * 60)
        report.append("【检测概况】")
        report.append("━" * 60)
        report.append(f"  总图片数：{total_images} 张")
        report.append(f"  成功检测：{success_count} 张")
        report.append(f"  检测失败：{failed_count} 张")
        report.append(f"  成功率：  {success_count/total_images*100:.1f}%")
        report.append("")
        
        report.append("━" * 60)
        report.append("【损伤统计】")
        report.append("━" * 60)
        report.append(f"  检测到的损伤总数：{total_detections} 个")
        report.append("")
        
        if class_stats:
            report.append("  损伤类型分布：")
            for class_name, stats in sorted(class_stats.items(), key=lambda x: x[1]["count"], reverse=True):
                report.append(f"    • {class_name}")
                report.append(f"      - 数量：{stats['count']} 个")
                report.append(f"      - 出现在 {stats['images']} 张图片中")
                report.append(f"      - 平均每张：{stats['count']/stats['images']:.2f} 个")
                report.append("")
        else:
            report.append("  未检测到任何损伤")
            report.append("")
        
        report.append("━" * 60)
        report.append("【详细结果】")
        report.append("━" * 60)
        
        # 按图片逐个显示
        for idx, result in enumerate(results, 1):
            filename = result.get("filename", f"图片{idx}")
            detection_time = result.get("detection_time_ms", 0)
            detections = result.get("detections", {})
            total_count = detections.get("total_count", 0)
            class_counts = detections.get("class_counts", {})
            detection_list = detections.get("detections", [])
            
            report.append(f"\n[{idx}] {filename}")
            report.append(f"    检测耗时：{detection_time:.1f} ms")
            report.append(f"    检测数量：{total_count} 个")
            
            if class_counts:
                report.append(f"    损伤类型：")
                for class_name, count in class_counts.items():
                    report.append(f"      - {class_name}: {count} 个")
            else:
                report.append(f"    结果：未检测到损伤")
            
            # 显示每个检测对象的详细信息
            if detection_list:
                report.append(f"    详细信息：")
                for det_idx, det in enumerate(detection_list, 1):
                    class_name = det.get("class_name", "未知")
                    confidence = det.get("confidence", 0)
                    bbox = det.get("bbox", {})
                    
                    report.append(f"      损伤#{det_idx} - {class_name}")
                    report.append(f"        置信度：{confidence:.2%}")
                    report.append(f"        位置：({bbox.get('x1', 0)}, {bbox.get('y1', 0)}) 至 ({bbox.get('x2', 0)}, {bbox.get('y2', 0)})")
                    report.append(f"        大小：{bbox.get('width', 0)} × {bbox.get('height', 0)} 像素")
        
        report.append("")
        report.append("=" * 60)
        report.append("                    报告结束")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def analyze_video_results(self) -> str:
        """分析视频检测结果"""
        if "video_info" not in self.data:
            return self._analyze_single_image()
        
        video_info = self.data["video_info"]
        frames = self.data.get("frames", [])
        
        # 视频信息
        filename = video_info.get("filename", "未知")
        total_frames = video_info.get("total_frames", len(frames))
        fps = video_info.get("fps", 0)
        duration = video_info.get("duration", 0)
        resolution = video_info.get("resolution", "未知")
        total_detections = video_info.get("total_detections", 0)
        
        # 统计数据
        class_stats = {}
        frames_with_detection = 0
        
        for frame in frames:
            detections = frame.get("detections", {})
            total_count = detections.get("total_count", 0)
            
            if total_count > 0:
                frames_with_detection += 1
            
            class_counts = detections.get("class_counts", {})
            for class_name, count in class_counts.items():
                if class_name not in class_stats:
                    class_stats[class_name] = {"total": 0, "frames": 0, "max_in_frame": 0}
                class_stats[class_name]["total"] += count
                class_stats[class_name]["frames"] += 1
                class_stats[class_name]["max_in_frame"] = max(class_stats[class_name]["max_in_frame"], count)
        
        # 生成报告
        report = []
        report.append("=" * 60)
        report.append("      太赫兹成像内部探伤检测 - 视频分析报告")
        report.append("=" * 60)
        report.append("")
        
        report.append("━" * 60)
        report.append("【视频信息】")
        report.append("━" * 60)
        report.append(f"  文件名：{filename}")
        report.append(f"  分辨率：{resolution}")
        report.append(f"  帧率：  {fps:.2f} FPS")
        report.append(f"  时长：  {duration:.2f} 秒")
        report.append(f"  总帧数：{total_frames} 帧")
        report.append("")
        
        report.append("━" * 60)
        report.append("【检测概况】")
        report.append("━" * 60)
        report.append(f"  检测到的损伤总数：{total_detections} 个")
        report.append(f"  有损伤的帧数：{frames_with_detection} 帧")
        report.append(f"  无损伤的帧数：{total_frames - frames_with_detection} 帧")
        report.append(f"  损伤检出率：{frames_with_detection/total_frames*100:.1f}%")
        report.append(f"  平均每帧损伤：{total_detections/total_frames:.2f} 个")
        report.append("")
        
        report.append("━" * 60)
        report.append("【损伤类型统计】")
        report.append("━" * 60)
        
        if class_stats:
            for class_name, stats in sorted(class_stats.items(), key=lambda x: x[1]["total"], reverse=True):
                report.append(f"\n  {class_name}：")
                report.append(f"    总数量：{stats['total']} 个")
                report.append(f"    出现帧数：{stats['frames']} 帧")
                report.append(f"    占比：{stats['frames']/total_frames*100:.1f}% 的帧包含此类型")
                report.append(f"    平均每帧：{stats['total']/stats['frames']:.2f} 个")
                report.append(f"    单帧最多：{stats['max_in_frame']} 个")
        else:
            report.append("  视频中未检测到任何损伤")
        
        report.append("")
        report.append("━" * 60)
        report.append("【时间轴分析】")
        report.append("━" * 60)
        
        # 找出损伤最多的帧
        max_detection_frame = None
        max_detection_count = 0
        
        for frame in frames:
            total_count = frame.get("detections", {}).get("total_count", 0)
            if total_count > max_detection_count:
                max_detection_count = total_count
                max_detection_frame = frame
        
        if max_detection_frame:
            frame_num = max_detection_frame.get("frame_number", 0)
            timestamp = max_detection_frame.get("timestamp", 0)
            detections_in_frame = max_detection_frame.get("detections", {}).get("detections", [])
            
            report.append(f"  损伤最多的帧：")
            report.append(f"    帧号：{frame_num}")
            report.append(f"    时间：{timestamp:.2f} 秒")
            report.append(f"    损伤数：{max_detection_count} 个")
            
            # 显示该帧的详细损伤信息
            if detections_in_frame:
                report.append(f"    该帧的损伤详情：")
                for det_idx, det in enumerate(detections_in_frame, 1):
                    class_name = det.get("class_name", "未知")
                    confidence = det.get("confidence", 0)
                    bbox = det.get("bbox", {})
                    
                    report.append(f"      #{det_idx} {class_name}")
                    report.append(f"        置信度：{confidence:.2%}")
                    report.append(f"        位置：({bbox.get('x1', 0)}, {bbox.get('y1', 0)}) - ({bbox.get('x2', 0)}, {bbox.get('y2', 0)})")
                    report.append(f"        大小：{bbox.get('width', 0)} × {bbox.get('height', 0)} px")
        
        report.append("")
        report.append("=" * 60)
        report.append("                    报告结束")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def _analyze_single_image(self) -> str:
        """分析单张图片检测结果"""
        total_count = self.data.get("total_count", 0)
        class_counts = self.data.get("class_counts", {})
        detections = self.data.get("detections", [])
        
        report = []
        report.append("=" * 60)
        report.append("      太赫兹成像内部探伤检测 - 图片分析报告")
        report.append("=" * 60)
        report.append("")
        
        report.append("━" * 60)
        report.append("【检测结果】")
        report.append("━" * 60)
        report.append(f"  检测到的损伤总数：{total_count} 个")
        report.append("")
        
        if class_counts:
            report.append("  损伤类型分布：")
            for class_name, count in class_counts.items():
                report.append(f"    • {class_name}: {count} 个")
        else:
            report.append("  未检测到损伤")
        
        report.append("")
        
        if detections:
            report.append("━" * 60)
            report.append("【详细信息】")
            report.append("━" * 60)
            
            for idx, det in enumerate(detections, 1):
                class_name = det.get("class_name", "未知")
                confidence = det.get("confidence", 0)
                bbox = det.get("bbox", {})
                
                report.append(f"\n  损伤 #{idx}：")
                report.append(f"    类型：{class_name}")
                report.append(f"    置信度：{confidence:.2%}")
                report.append(f"    位置：X={bbox.get('x1', 0)}, Y={bbox.get('y1', 0)}")
                report.append(f"    大小：{bbox.get('width', 0)} x {bbox.get('height', 0)} 像素")
        
        report.append("")
        report.append("=" * 60)
        report.append("                    报告结束")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def generate_report(self, output_path: str = None) -> str:
        """生成并保存报告"""
        # 判断数据类型
        if "batch_info" in self.data:
            report = self.analyze_batch_results()
        elif "video_info" in self.data:
            report = self.analyze_video_results()
        else:
            report = self._analyze_single_image()
        
        # 保存报告
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
    
    def generate_summary(self) -> Dict:
        """生成简要摘要（用于界面显示）"""
        if "batch_info" in self.data:
            return self._summary_batch()
        elif "video_info" in self.data:
            return self._summary_video()
        else:
            return self._summary_single()
    
    def _summary_batch(self) -> Dict:
        """批量检测摘要"""
        batch_info = self.data.get("batch_info", {})
        results = self.data.get("results", [])
        
        total_detections = sum(
            r.get("detections", {}).get("total_count", 0) 
            for r in results
        )
        
        return {
            "type": "批量图片检测",
            "total_images": batch_info.get("total_images", 0),
            "success": batch_info.get("success", 0),
            "total_detections": total_detections,
            "confidence": batch_info.get("confidence_threshold", 0.5)
        }
    
    def _summary_video(self) -> Dict:
        """视频检测摘要"""
        video_info = self.data.get("video_info", {})
        return {
            "type": "视频检测",
            "filename": video_info.get("filename", ""),
            "total_frames": video_info.get("total_frames", 0),
            "total_detections": video_info.get("total_detections", 0),
            "fps": video_info.get("fps", 0),
            "duration": video_info.get("duration", 0)
        }
    
    def _summary_single(self) -> Dict:
        """单图片摘要"""
        return {
            "type": "单图片检测",
            "total_detections": self.data.get("total_count", 0),
            "classes": self.data.get("class_counts", {})
        }


def convert_json_to_report(json_path: str, output_path: str = None) -> str:
    """
    将JSON检测结果转换为可读报告
    
    Args:
        json_path: JSON文件路径
        output_path: 报告输出路径（可选）
    
    Returns:
        str: 生成的报告文本
    """
    analyzer = DetectionAnalyzer(json_path)
    
    if output_path is None:
        # 自动生成报告文件名
        json_file = Path(json_path)
        output_path = str(json_file.parent / f"{json_file.stem}_报告.txt")
    
    report = analyzer.generate_report(output_path)
    return report


if __name__ == "__main__":
    # 测试用例
    import sys
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        report = convert_json_to_report(json_file)
        print(report)
    else:
        print("使用方法：python json_analyzer.py <json文件路径>")


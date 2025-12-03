# 猪打架行为检测与评估系统

基于YOLO目标跟踪的打架行为检测系统，支持滑动窗口检测和自动化性能评估。

## 功能特性

1. **目标跟踪**: 使用YOLO + BoTSORT/ByteTrack进行多目标跟踪
2. **打架检测**: 基于滑动窗口的打架行为识别
3. **性能评估**: 与Ground Truth对比，输出精确率、召回率、F1分数等指标
4. **批量处理**: 支持批量处理多个视频并生成评估报告

## 文件说明

- `track_with_fight_detection.py`: 单视频检测与评估脚本
- `batch_evaluate.py`: 批量视频评估脚本
- `ground_truth_example.json`: Ground Truth文件格式示例
- `track.py`: 原始的跟踪脚本（保留）

## 安装依赖

```bash
pip install ultralytics opencv-python numpy
```

## 使用方法

### 1. 单视频检测

```bash
python track_with_fight_detection.py \
    --weights your_model.pt \
    --source test_video.mp4 \
    --window-size 30 \
    --stride 15 \
    --distance-threshold 100 \
    --speed-threshold 50
```

### 2. 单视频检测 + 评估

```bash
python track_with_fight_detection.py \
    --weights your_model.pt \
    --source test_video.mp4 \
    --gt-file ground_truth.json \
    --eval-iou-threshold 0.5 \
    --output-dir results
```

### 3. 批量评估

```bash
python batch_evaluate.py \
    --weights your_model.pt \
    --video-dir ./test_videos \
    --gt-file ground_truth.json \
    --output-dir batch_results \
    --window-size 30 \
    --stride 15
```

## Ground Truth 文件格式

创建一个JSON文件，格式如下：

### 方式1: 使用帧号（推荐）

```json
{
  "video1.mp4": [
    [50, 180],
    [300, 450]
  ],
  "video2.mp4": [
    [100, 250]
  ]
}
```

### 方式2: 使用时间（秒）

```json
{
  "fps": 30,
  "video1.mp4": [
    [1.5, 6.0],
    [10.0, 15.0]
  ],
  "video2.mp4": [
    [3.3, 8.5]
  ]
}
```

- 每个视频对应一个片段列表
- 每个片段是 `[开始帧/时间, 结束帧/时间]`
- 如果使用时间（秒），需要提供 `fps` 字段

## 参数说明

### 跟踪相关参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | - | 模型权重文件路径 |
| `--source` | - | 视频源（必需） |
| `--tracker` | botsort.yaml | 跟踪器配置 |
| `--device` | 0 | GPU设备编号 |
| `--conf` | 0.75 | 检测置信度阈值 |
| `--iou` | 0.6 | NMS IoU阈值 |
| `--imgsz` | 640 | 推理图像尺寸 |

### 打架检测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--window-size` | 30 | 滑动窗口大小（帧数） |
| `--stride` | 15 | 窗口滑动步长 |
| `--distance-threshold` | 100 | 判断距离过近的阈值（像素） |
| `--speed-threshold` | 50 | 判断运动剧烈的速度阈值 |
| `--min-fight-duration` | 15 | 最小打架持续帧数 |

### 评估参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gt-file` | None | Ground Truth JSON文件 |
| `--eval-iou-threshold` | 0.5 | 评估时IoU匹配阈值 |
| `--output-dir` | fight_detection_results | 输出目录 |

## 检测原理

### 打架行为判断依据

系统通过以下特征判断打架行为：

1. **距离特征**: 检测两只或多只猪的中心点距离
   - 距离小于 `distance_threshold` 时记为"接近"
   - 计算窗口内接近的比例

2. **运动特征**: 跟踪每只猪的移动速度
   - 计算帧间位移
   - 速度超过 `speed_threshold` 时记为"剧烈运动"

3. **综合判断**: 结合距离和运动特征
   - 计算综合置信度分数
   - 分数 > 0.5 判定为打架

### 滑动窗口机制

```
视频帧序列: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]

Window 1:   [0, 1, 2, ..., 29]         <- 检测
Window 2:              [15, 16, ..., 44]       <- 检测
Window 3:                     [30, 31, ..., 59]       <- 检测
...
```

- 窗口大小: `window_size` 帧
- 滑动步长: `stride` 帧
- 重叠窗口检测到的片段会自动合并

## 评估指标

系统计算以下指标：

- **Precision (精确率)**: TP / (TP + FP)
  - 预测为打架的片段中，真正是打架的比例
  
- **Recall (召回率)**: TP / (TP + FN)
  - 实际打架片段中，被正确检测出的比例
  
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
  - 精确率和召回率的调和平均数

- **IoU匹配**: 预测片段与GT片段的时间重叠度
  - IoU ≥ threshold 时视为匹配成功

## 输出文件

### 单视频模式

```
fight_detection_results/
├── test_video_predictions.json    # 预测结果
└── test_video_evaluation.json     # 评估指标（如果提供GT）
```

### 批量模式

```
batch_evaluation_results/
├── overall_evaluation.json        # 总体评估结果（JSON）
└── evaluation_report.txt          # 详细评估报告（文本）
```

## 参数调优建议

### 1. 窗口参数

- **window_size**: 
  - 太小：可能遗漏持续时间较长的打架
  - 太大：时间分辨率低，定位不精确
  - 建议: 根据平均打架时长设置（通常30-60帧）

- **stride**:
  - 太小：计算量大，但检测更细致
  - 太大：可能遗漏短暂的打架
  - 建议: window_size 的 1/2 或 1/3

### 2. 距离阈值 (distance_threshold)

- 根据视频分辨率和猪的大小调整
- 建议先在几个样本上可视化，观察打架时的距离
- 典型值: 50-150像素（640×480视频）

### 3. 速度阈值 (speed_threshold)

- 根据视频帧率和猪的运动速度调整
- 帧率越高，帧间位移越小
- 建议先计算正常运动和打架时的平均速度

### 4. 最小持续时长 (min_fight_duration)

- 过滤掉太短的噪声片段
- 建议: 0.5-1秒对应的帧数

## 示例工作流程

### 步骤1: 准备Ground Truth

创建 `ground_truth.json`:

```json
{
  "fps": 30,
  "pig_fight_001.mp4": [[2.5, 8.0], [15.0, 22.3]],
  "pig_fight_002.mp4": [[5.1, 12.8]]
}
```

### 步骤2: 参数调优（单视频测试）

```bash
python track_with_fight_detection.py \
    --weights best.pt \
    --source pig_fight_001.mp4 \
    --gt-file ground_truth.json \
    --window-size 30 \
    --stride 10 \
    --distance-threshold 80 \
    --show
```

观察结果，调整参数直到满意。

### 步骤3: 批量评估

```bash
python batch_evaluate.py \
    --weights best.pt \
    --video-dir ./test_videos \
    --gt-file ground_truth.json \
    --window-size 30 \
    --stride 10 \
    --distance-threshold 80 \
    --output-dir final_results
```

### 步骤4: 分析结果

查看 `final_results/evaluation_report.txt`，分析各视频表现。

## 常见问题

### Q1: 误检率高怎么办？

- 提高 `conf` 阈值（减少误检目标）
- 增大 `distance_threshold`（更严格的接近条件）
- 增大 `speed_threshold`（更剧烈的运动才算）
- 增大 `min_fight_duration`（过滤短片段）

### Q2: 漏检率高怎么办？

- 降低 `conf` 阈值（检测更多目标）
- 减小 `distance_threshold`
- 减小 `speed_threshold`
- 减小 `stride`（更密集的窗口）
- 增大 `window_size`（捕捉更长的行为）

### Q3: 检测片段时间不准确？

- 减小 `stride`（更细的时间粒度）
- 调整 `eval_iou_threshold`（更宽松的匹配）

### Q4: 需要自定义打架判断逻辑？

修改 `FightDetector` 类的 `is_fighting_in_window` 方法，可以：
- 添加更多特征（如目标框面积变化、方向改变等）
- 调整特征权重
- 使用机器学习模型替代规则判断

## 高级功能扩展

### 1. 添加更多特征

```python
# 在 FightDetector.is_fighting_in_window 中添加:

# 特征3: 检测框面积变化（可能的遮挡）
area_changes = []
for i in range(1, len(window_frames)):
    # ... 计算面积变化率
    
# 特征4: 检测框长宽比变化
aspect_ratio_changes = []
# ...
```

### 2. 可视化检测结果

```python
# 在检测到的片段上标注
import cv2

cap = cv2.VideoCapture(video_path)
for start, end, conf in fight_segments:
    # 读取并标注帧
    # 保存标注视频
```

### 3. 导出为其他格式

可以将结果导出为：
- CSV格式（便于Excel分析）
- 视频剪辑（自动裁剪打架片段）
- 可视化图表（时间轴标注）

## 性能优化

- 使用 `--vid-stride > 1` 跳帧处理（降低精度但加速）
- 使用较小的 `--imgsz` 推理尺寸
- 使用 GPU 加速（`--device 0`）
- 批量处理时设置 `--save False` 不保存可视化

## 引用与致谢

本系统基于以下开源项目：
- Ultralytics YOLO
- BoTSORT / ByteTrack

---

**作者**: Assistant  
**更新日期**: 2024-11  
**版本**: 1.0


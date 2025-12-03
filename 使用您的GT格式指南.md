# 使用您的Ground Truth格式指南

## 📋 您的GT格式说明

您的Ground Truth文件使用的是 **database格式**，结构如下：

```json
{
  "database": {
    "MyVideo_191": {
      "subset": "test",
      "annotations": [
        {"segment": ["93.1", "98.2"], "label": "fight"},
        {"segment": ["101.6", "114.0"], "label": "fight"},
        {"segment": ["137.5", "143.1"], "label": "fight"}
      ]
    },
    "MyVideo_192": {
      "subset": "test",
      "annotations": [
        {"segment": ["45.0", "53.0"], "label": "fight"},
        {"segment": ["61.8", "65.1"], "label": "fight"}
      ]
    }
  }
}
```

**格式特点**:
- 外层有 `database` 键
- 每个视频有 `subset` 和 `annotations` 字段
- `annotations` 是一个列表，包含多个标注
- 每个标注有 `segment`（时间段，**秒为单位**）和 `label`（标签）
- 只会提取 `label: "fight"` 的片段

---

## ✅ 系统已自动适配

**好消息！** 我已经更新了所有代码，系统现在自动支持您的GT格式。无需手动转换。

### 自动识别机制

系统会自动检测GT文件格式：
1. 如果有 `database` 键 → 使用database格式解析
2. 如果有 `fps` 键 → 使用简单时间格式
3. 否则 → 使用简单帧号格式

---

## 🚀 直接使用示例

### 1. 单视频检测与评估

```bash
python track_with_fight_detection.py \
    --weights runs/train/v10-APConv-AssemFormer-HSFPN-ATFLm_exp/weights/best.pt \
    --source MyVideo_191.mp4 \
    --gt-file ground_truth_example.json \
    --fps 30 \
    --window-size 30 \
    --stride 15
```

**注意事项**:
- `--source` 的视频文件名需要与GT中的视频名称匹配
- 例如：GT中是 `MyVideo_191`，则视频文件可以是：
  - `MyVideo_191.mp4`
  - `MyVideo_191.avi`
  - `MyVideo_191.mov`
  - 等等（只要视频名部分匹配即可）

### 2. 批量评估

```bash
python batch_evaluate.py \
    --weights best.pt \
    --video-dir ./test_videos \
    --gt-file ground_truth_example.json \
    --fps 30 \
    --window-size 30 \
    --stride 15 \
    --output-dir results
```

**视频文件夹结构**:
```
test_videos/
├── MyVideo_191.mp4
├── MyVideo_192.mp4
├── MyVideo_193.mp4
└── ...
```

### 3. 可视化结果

```bash
python visualize_results.py \
    --video test_videos/MyVideo_191.mp4 \
    --pred results/MyVideo_191_predictions.json \
    --gt ground_truth_example.json \
    --fps 30
```

---

## ⚙️ 重要参数说明

### `--fps` 参数

由于您的GT中的时间是**秒为单位**（如 `"93.1"` 表示93.1秒），系统需要知道帧率才能转换为帧号。

**设置方法1**: 命令行指定（推荐）
```bash
--fps 30  # 如果视频是30fps
```

**设置方法2**: 在GT文件中添加
```json
{
  "fps": 30,
  "database": {
    "MyVideo_191": { ... }
  }
}
```

**如何确定视频的实际fps?**

```bash
# 方法1: 使用ffprobe
ffprobe -v quiet -show_streams MyVideo_191.mp4 | grep r_frame_rate

# 方法2: 使用Python
import cv2
cap = cv2.VideoCapture('MyVideo_191.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps}")
```

---

## 📊 视频名称匹配规则

系统会**自动去除文件扩展名**后进行匹配：

| GT中的键 | 视频文件名 | 匹配结果 | 说明 |
|---------|-----------|---------|------|
| `MyVideo_191` | `MyVideo_191.mp4` | ✓ | 自动去除.mp4后匹配 |
| `MyVideo_191` | `MyVideo_191.avi` | ✓ | 自动去除.avi后匹配 |
| `MyVideo_192` | `MyVideo_192.mov` | ✓ | 自动去除.mov后匹配 |
| `MyVideo_193` | `test_MyVideo_193.mp4` | ✗ | 去除扩展名后仍不匹配 |
| `MyVideo_194` | `MyVideo_194` | ✓ | 即使无扩展名也可匹配 |

**匹配逻辑**: 
1. 系统会取视频文件名的主干部分（不含扩展名）
2. 用这个主干名称在GT的database中查找
3. 因此您的GT中只需要 `"MyVideo_191"` 即可，无需包含 `.mp4` 等扩展名

**建议**: 视频文件的主文件名（不含扩展名）与GT键名完全一致

---

## 🎯 完整工作流程

### 步骤1: 准备视频文件

确保视频文件名与GT中的键匹配：

```bash
# 如果视频文件名不匹配，可以批量重命名
# 例如：video_191.mp4 -> MyVideo_191.mp4

# Linux/Mac:
for f in video_*.mp4; do
    num=$(echo $f | grep -oP '\d+')
    mv "$f" "MyVideo_${num}.mp4"
done

# Windows PowerShell:
Get-ChildItem video_*.mp4 | ForEach-Object {
    $num = $_.Name -replace '\D+', ''
    Rename-Item $_ -NewName "MyVideo_$num.mp4"
}
```

### 步骤2: 确定视频帧率

```bash
# 检查一个视频的帧率
ffprobe -v quiet -show_streams test_videos/MyVideo_191.mp4 | grep r_frame_rate
# 输出类似: r_frame_rate=30/1  (表示30fps)
```

如果所有视频帧率相同，在GT文件中添加fps：

```json
{
  "fps": 30,
  "database": { ... }
}
```

### 步骤3: 单视频测试

选择一个视频先测试：

```bash
python track_with_fight_detection.py \
    --weights your_model.pt \
    --source test_videos/MyVideo_191.mp4 \
    --gt-file ground_truth_example.json \
    --fps 30 \
    --show
```

查看输出：
```
加载Ground Truth: ground_truth_example.json
GT包含 3 个打架片段

检测到 X 个打架片段:
  片段 1: 帧 XXX-XXX (置信度: X.XXX)
  ...

评估结果:
精确率 (Precision): X.XXXX
召回率 (Recall):    X.XXXX
F1分数 (F1-Score):  X.XXXX
```

### 步骤4: 可视化验证

```bash
python visualize_results.py \
    --video test_videos/MyVideo_191.mp4 \
    --pred fight_detection_results/MyVideo_191_predictions.json \
    --gt ground_truth_example.json \
    --fps 30
```

**查看要点**:
- 绿色条：GT标注的打架片段
- 红色条：系统预测的打架片段
- 白线：当前播放位置
- 屏幕右上角显示：TP/FP/FN状态

### 步骤5: 调整参数

根据可视化结果调整参数：

```bash
# 如果误检多（很多红色不在绿色上）
--distance-threshold 120  # 提高
--speed-threshold 60      # 提高
--min-fight-duration 20   # 提高

# 如果漏检多（很多绿色没有红色）
--distance-threshold 80   # 降低
--speed-threshold 40      # 降低
--stride 10               # 减小步长
```

### 步骤6: 批量评估

参数满意后，评估所有视频：

```bash
python batch_evaluate.py \
    --weights your_model.pt \
    --video-dir ./test_videos \
    --gt-file ground_truth_example.json \
    --fps 30 \
    --window-size 30 \
    --stride 15 \
    --distance-threshold 100 \
    --speed-threshold 50 \
    --output-dir final_results
```

### 步骤7: 查看报告

```bash
# 查看文本报告
cat final_results/evaluation_report.txt

# 或在Python中分析JSON
python
>>> import json
>>> with open('final_results/overall_evaluation.json') as f:
...     data = json.load(f)
>>> print(f"总体F1: {data['overall_metrics']['f1']:.4f}")
```

---

## 🔍 常见问题

### Q1: 提示"GT中未找到视频"

**原因**: 视频文件名与GT键名不匹配

**解决**:
```python
# 检查GT中有哪些视频
import json
with open('ground_truth_example.json') as f:
    data = json.load(f)
    
print("GT中的视频列表:")
for video_name in data['database'].keys():
    print(f"  - {video_name}")

# 检查视频文件夹中的文件
import os
print("\n实际视频文件:")
for f in os.listdir('test_videos'):
    if f.endswith(('.mp4', '.avi', '.mov')):
        print(f"  - {f}")
```

对比后重命名视频文件使其匹配。

### Q2: 时间对不上，检测结果偏移

**原因**: fps设置不正确

**解决**:
1. 检查实际视频fps
2. 确保 `--fps` 参数与实际一致
3. 如果不同视频fps不同，需要分别指定

### Q3: 所有视频都检测不到打架

**原因**: 
1. 跟踪效果不好
2. 参数阈值设置不合理

**解决**:
```bash
# 1. 先测试跟踪
python track.py \
    --source test_videos/MyVideo_191.mp4 \
    --weights your_model.pt \
    --show

# 观察：能否稳定跟踪到猪？ID是否频繁变化？

# 2. 大幅降低阈值测试
python track_with_fight_detection.py \
    --source test_videos/MyVideo_191.mp4 \
    --gt-file ground_truth_example.json \
    --distance-threshold 50 \
    --speed-threshold 20 \
    --min-fight-duration 5
```

### Q4: 如何处理不同视频有不同fps?

**方法1**: 在GT中为每个视频指定fps（需要修改GT格式）

**方法2**: 分组处理
```bash
# 30fps的视频
python batch_evaluate.py \
    --video-dir ./test_videos_30fps \
    --gt-file ground_truth.json \
    --fps 30 \
    --output-dir results_30fps

# 25fps的视频
python batch_evaluate.py \
    --video-dir ./test_videos_25fps \
    --gt-file ground_truth.json \
    --fps 25 \
    --output-dir results_25fps
```

---

## 💡 高级技巧

### 技巧1: 批量提取视频信息

创建 `extract_video_info.py`:

```python
import cv2
import json
from pathlib import Path

video_dir = Path('test_videos')
video_info = {}

for video_file in video_dir.glob('*.mp4'):
    cap = cv2.VideoCapture(str(video_file))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    video_info[video_file.stem] = {
        'fps': fps,
        'total_frames': total_frames,
        'duration': duration
    }
    cap.release()
    
    print(f"{video_file.name}: {fps:.2f} fps, {duration:.2f}s")

with open('video_info.json', 'w') as f:
    json.dump(video_info, f, indent=2)
```

### 技巧2: 验证GT时间是否在视频范围内

```python
import json
import cv2
from pathlib import Path

with open('ground_truth_example.json') as f:
    gt = json.load(f)

fps = 30  # 或从gt['fps']读取

for video_name, video_data in gt['database'].items():
    video_file = Path(f'test_videos/{video_name}.mp4')
    
    if not video_file.exists():
        print(f"警告: 视频文件不存在 - {video_name}")
        continue
    
    cap = cv2.VideoCapture(str(video_file))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    
    print(f"\n{video_name}:")
    print(f"  视频时长: {duration:.1f}秒")
    
    for ann in video_data['annotations']:
        if ann['label'] == 'fight':
            start, end = float(ann['segment'][0]), float(ann['segment'][1])
            
            if end > duration:
                print(f"  ⚠️ 标注超出范围: [{start:.1f}, {end:.1f}]秒 > {duration:.1f}秒")
            else:
                print(f"  ✓ [{start:.1f}, {end:.1f}]秒")
```

### 技巧3: 统计GT信息

```python
import json

with open('ground_truth_example.json') as f:
    gt = json.load(f)

total_videos = len(gt['database'])
total_segments = 0
total_duration = 0
fps = 30

print(f"总视频数: {total_videos}")

for video_name, video_data in gt['database'].items():
    segments = [ann for ann in video_data['annotations'] if ann['label'] == 'fight']
    total_segments += len(segments)
    
    for seg in segments:
        start, end = float(seg['segment'][0]), float(seg['segment'][1])
        total_duration += (end - start)

print(f"总打架片段数: {total_segments}")
print(f"总打架时长: {total_duration:.1f}秒 ({total_duration/60:.1f}分钟)")
print(f"平均每视频: {total_segments/total_videos:.1f}个片段")
print(f"平均片段时长: {total_duration/total_segments:.1f}秒")
```

---

## 📝 快速参考命令

```bash
# 单视频测试
python track_with_fight_detection.py \
    --weights best.pt \
    --source test_videos/MyVideo_191.mp4 \
    --gt-file ground_truth_example.json \
    --fps 30 \
    --show

# 批量评估
python batch_evaluate.py \
    --weights best.pt \
    --video-dir ./test_videos \
    --gt-file ground_truth_example.json \
    --fps 30 \
    --output-dir results

# 可视化
python visualize_results.py \
    --video test_videos/MyVideo_191.mp4 \
    --pred results/MyVideo_191_predictions.json \
    --gt ground_truth_example.json \
    --fps 30
```

---

**现在您可以直接使用您的GT文件了！祝评估顺利！** 🎉


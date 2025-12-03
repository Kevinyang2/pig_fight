# 快速入门指南

## 🚀 5分钟上手

### 第一步：准备Ground Truth

使用交互式工具标注打架片段：

```bash
python create_ground_truth.py --video test_video.mp4 --mode interactive --output gt.json
```

或手动输入：

```bash
python create_ground_truth.py --video test_video.mp4 --mode manual --output gt.json
```

批量标注多个视频：

```bash
python create_ground_truth.py --video-dir ./test_videos --mode interactive --output gt.json
```

### 第二步：运行检测

单个视频检测+评估：

```bash
python track_with_fight_detection.py \
    --weights runs/train/v10-APConv-AssemFormer-HSFPN-ATFLm_exp/weights/best.pt \
    --source test_video.mp4 \
    --gt-file gt.json \
    --window-size 30 \
    --stride 15
```

### 第三步：查看结果

可视化检测结果：

```bash
python visualize_results.py \
    --video test_video.mp4 \
    --pred fight_detection_results/test_video_predictions.json \
    --gt gt.json
```

保存可视化视频：

```bash
python visualize_results.py \
    --video test_video.mp4 \
    --pred fight_detection_results/test_video_predictions.json \
    --gt gt.json \
    --output visualized_output.mp4
```

### 第四步：批量评估（可选）

如果有多个测试视频：

```bash
python batch_evaluate.py \
    --weights your_model.pt \
    --video-dir ./test_videos \
    --gt-file gt.json \
    --output-dir batch_results
```

查看 `batch_results/evaluation_report.txt` 获取详细报告。

---

## 📊 结果示例

### 终端输出

```
检测到 3 个打架片段:
  片段 1: 帧 50-180 (置信度: 0.753)
  片段 2: 帧 300-450 (置信度: 0.821)
  片段 3: 帧 600-750 (置信度: 0.692)

==================================================
评估结果:
==================================================
精确率 (Precision): 0.8571
召回率 (Recall):    0.7500
F1分数 (F1-Score):  0.8000
真正例 (TP):        3
假正例 (FP):        0
假负例 (FN):        1
IoU阈值:            0.5
==================================================
```

### JSON输出文件

`test_video_predictions.json`:
```json
{
  "video": "test_video.mp4",
  "segments": [
    [50, 180, 0.753],
    [300, 450, 0.821],
    [600, 750, 0.692]
  ],
  "total_frames": 900
}
```

`test_video_evaluation.json`:
```json
{
  "video": "test_video.mp4",
  "metrics": {
    "precision": 0.8571,
    "recall": 0.7500,
    "f1": 0.8000,
    "tp": 3,
    "fp": 0,
    "fn": 1
  },
  "predictions": [[50, 180, 0.753], [300, 450, 0.821], [600, 750, 0.692]],
  "ground_truth": [[45, 175], [295, 455], [580, 720], [850, 890]]
}
```

---

## 🎯 参数调优技巧

### 问题：太多误检（FP高）

**方案1: 提高检测门槛**
```bash
--distance-threshold 120  # 默认100，提高后更严格
--speed-threshold 60      # 默认50，提高后更严格
--min-fight-duration 20   # 默认15，过滤更多短片段
```

**方案2: 调整置信度**
```bash
--conf 0.80  # 默认0.75，提高检测置信度
```

### 问题：漏检太多（FN高）

**方案1: 降低检测门槛**
```bash
--distance-threshold 80   # 降低
--speed-threshold 40      # 降低
--min-fight-duration 10   # 降低
```

**方案2: 增加窗口重叠**
```bash
--window-size 40  # 增大窗口
--stride 10       # 减小步长，增加重叠
```

### 问题：时间定位不准

```bash
--stride 5  # 大幅减小步长，提高时间分辨率
```

---

## 🔧 常见问题

### Q: 如何查看中间结果？

在跟踪时添加 `--show` 和 `--save` 参数：

```bash
python track_with_fight_detection.py \
    --source test.mp4 \
    --show \
    --save
```

### Q: 如何只看打架片段？

使用可视化工具，按空格键暂停，用 A/D 键快速浏览。

### Q: 如何导出Excel格式？

修改输出代码，或使用pandas：

```python
import json
import pandas as pd

with open('results.json') as f:
    data = json.load(f)

df = pd.DataFrame(data['segments'], columns=['start', 'end', 'confidence'])
df.to_excel('results.xlsx', index=False)
```

### Q: 如何修改打架判断逻辑？

编辑 `track_with_fight_detection.py` 中的 `FightDetector.is_fighting_in_window` 方法。

例如，添加目标数量条件：

```python
def is_fighting_in_window(self, start_idx: int, end_idx: int):
    # ... 原有代码 ...
    
    # 新增：至少要有2只猪
    avg_obj_count = np.mean([len(f['objects']) for f in window_frames])
    if avg_obj_count < 2:
        return False, 0.0
    
    # ... 继续原有逻辑 ...
```

---

## 📈 性能对比实验

测试不同参数组合：

```bash
# 组合1: 保守策略（高精确率）
python track_with_fight_detection.py --source test.mp4 --gt-file gt.json \
    --distance-threshold 120 --speed-threshold 60 --min-fight-duration 25

# 组合2: 激进策略（高召回率）
python track_with_fight_detection.py --source test.mp4 --gt-file gt.json \
    --distance-threshold 80 --speed-threshold 40 --min-fight-duration 10

# 组合3: 平衡策略
python track_with_fight_detection.py --source test.mp4 --gt-file gt.json \
    --distance-threshold 100 --speed-threshold 50 --min-fight-duration 15
```

记录每组的 Precision、Recall、F1，选择最佳组合。

---

## 💡 进阶用法

### 1. 导出打架片段视频

```python
import cv2
import json

with open('predictions.json') as f:
    data = json.load(f)

cap = cv2.VideoCapture(data['video'])
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

for i, (start, end, conf) in enumerate(data['segments']):
    writer = cv2.VideoWriter(f'fight_segment_{i+1}.mp4', fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for frame_idx in range(start, end + 1):
        ret, frame = cap.read()
        if ret:
            writer.write(frame)
    
    writer.release()
```

### 2. 生成热力图

统计每一帧的打架概率（滑动窗口置信度平均）并可视化。

### 3. 多阈值评估

批量测试不同IoU阈值下的性能曲线。

---

## 📞 需要帮助？

- 查看完整文档: `README_fight_detection.md`
- 示例GT文件: `ground_truth_example.json`
- 检查代码中的注释和docstring


# 快速开始 - 使用您的GT格式

## 🎯 5分钟快速测试

### 第1步：测试GT文件是否正确

```bash
python test_gt_loading.py \
    --gt-file ground_truth_example.json \
    --video-dir ./test_videos \
    --fps 30
```

**输出示例**:
```
==================================================
Ground Truth 文件验证工具
==================================================

[步骤1] 加载GT文件: ground_truth_example.json
✓ GT文件加载成功

[步骤2] 检测GT格式
✓ 检测到database格式
  - FPS: 30.0 (从参数获取)
  - 视频数量: 43

[步骤3] GT内容概览
视频名称              片段数     总时长(秒)
------------------------------------------------------------
MyVideo_191              3         26.6
MyVideo_192              2         15.3
...
------------------------------------------------------------
总计                   XXX       XXXX.X

[步骤4] 验证与视频文件的匹配
✓ MyVideo_191        180.5s  FPS: ✓
✓ MyVideo_192        120.3s  FPS: ✓
...
```

**如果有问题**:
- ✗ 文件不存在 → 检查路径
- ✗ JSON格式错误 → 检查JSON语法
- ⚠ 标注超出范围 → GT时间超过视频长度
- ⚠ FPS不匹配 → 检查实际视频fps

---

### 第2步：单视频测试

选择一个视频测试（例如 MyVideo_191）:

```bash
python track_with_fight_detection.py \
    --weights runs/train/v10-APConv-AssemFormer-HSFPN-ATFLm_exp/weights/best.pt \
    --source test_videos/MyVideo_191.mp4 \
    --gt-file ground_truth_example.json \
    --fps 30 \
    --show
```

**查看输出**:
```
加载Ground Truth: ground_truth_example.json
GT包含 3 个打架片段

开始跟踪: test_videos/MyVideo_191.mp4
已处理 100 帧
已处理 200 帧
...

检测到 X 个打架片段:
  片段 1: 帧 2793-2946 (置信度: 0.XXX)
  片段 2: 帧 3048-3420 (置信度: 0.XXX)
  片段 3: 帧 4125-4293 (置信度: 0.XXX)

==================================================
评估结果:
==================================================
精确率 (Precision): 0.XXXX
召回率 (Recall):    0.XXXX
F1分数 (F1-Score):  0.XXXX
真正例 (TP):        X
假正例 (FP):        X
假负例 (FN):        X
==================================================
```

---

### 第3步：可视化验证

```bash
python visualize_results.py \
    --video test_videos/MyVideo_191.mp4 \
    --pred fight_detection_results/MyVideo_191_predictions.json \
    --gt ground_truth_example.json \
    --fps 30
```

**观察**:
- **绿色条**: GT标注的打架片段
- **红色条**: 系统预测的打架片段
- **白色线**: 当前播放位置

**判断**:
- 红色和绿色重叠多 → 检测准确 ✓
- 红色多但绿色少 → 误检多（需要提高阈值）
- 绿色多但红色少 → 漏检多（需要降低阈值）

**控制键**:
- `空格` - 暂停/继续
- `A` - 后退10帧
- `D` - 前进10帧
- `Q` - 退出

---

### 第4步：批量评估（可选）

如果单视频效果满意，批量评估所有视频：

```bash
python batch_evaluate.py \
    --weights runs/train/v10-APConv-AssemFormer-HSFPN-ATFLm_exp/weights/best.pt \
    --video-dir ./test_videos \
    --gt-file ground_truth_example.json \
    --fps 30 \
    --output-dir batch_results
```

**查看结果**:
```bash
# 查看总体报告
cat batch_results/evaluation_report.txt

# 或查看JSON（可用Python处理）
python -c "
import json
with open('batch_results/overall_evaluation.json') as f:
    data = json.load(f)
print(f\"总体F1分数: {data['overall_metrics']['f1']:.4f}\")
"
```

---

## ⚙️ 重要提醒

### 1. 视频文件命名

系统会**自动去掉扩展名**进行匹配：

| GT中的键 | 视频文件名 | 匹配结果 |
|---------|-----------|---------|
| MyVideo_191 | MyVideo_191.mp4 | ✓ 自动匹配 |
| MyVideo_191 | MyVideo_191.avi | ✓ 自动匹配 |
| MyVideo_192 | MyVideo_192.mov | ✓ 自动匹配 |
| MyVideo_193 | video_193.mp4 | ✗ 名称不同 |

### 2. FPS设置

您的GT中时间是**秒为单位**，必须提供正确的fps：

```bash
# 方式1: 命令行指定（每次都要加）
--fps 30

# 方式2: 在GT文件中添加（推荐）
{
  "fps": 30,
  "database": { ... }
}
```

### 3. 检查视频实际fps

```bash
# 使用ffprobe
ffprobe -v quiet -show_streams test_videos/MyVideo_191.mp4 | grep r_frame_rate

# 输出: r_frame_rate=30/1  (表示30fps)
```

---

## 🔧 参数调优

根据第一次测试结果调整参数：

### 情况A: 误检太多（FP高）

```bash
python track_with_fight_detection.py \
    --source test_videos/MyVideo_191.mp4 \
    --gt-file ground_truth_example.json \
    --fps 30 \
    --distance-threshold 120 \
    --speed-threshold 60 \
    --min-fight-duration 20 \
    --conf 0.80
```

### 情况B: 漏检太多（FN高）

```bash
python track_with_fight_detection.py \
    --source test_videos/MyVideo_191.mp4 \
    --gt-file ground_truth_example.json \
    --fps 30 \
    --distance-threshold 80 \
    --speed-threshold 40 \
    --stride 10 \
    --conf 0.70
```

### 情况C: 时间定位不准

```bash
python track_with_fight_detection.py \
    --source test_videos/MyVideo_191.mp4 \
    --gt-file ground_truth_example.json \
    --fps 30 \
    --stride 5 \
    --window-size 40
```

---

## 📋 完整命令模板

### 单视频检测与评估

```bash
python track_with_fight_detection.py \
    --weights <你的模型.pt> \
    --source test_videos/<视频名>.mp4 \
    --gt-file ground_truth_example.json \
    --fps 30 \
    --device 0 \
    --conf 0.75 \
    --window-size 30 \
    --stride 15 \
    --distance-threshold 100 \
    --speed-threshold 50 \
    --min-fight-duration 15 \
    --show \
    --output-dir results
```

### 批量评估

```bash
python batch_evaluate.py \
    --weights <你的模型.pt> \
    --video-dir ./test_videos \
    --gt-file ground_truth_example.json \
    --fps 30 \
    --device 0 \
    --conf 0.75 \
    --window-size 30 \
    --stride 15 \
    --distance-threshold 100 \
    --speed-threshold 50 \
    --min-fight-duration 15 \
    --output-dir batch_results
```

### 可视化

```bash
python visualize_results.py \
    --video test_videos/<视频名>.mp4 \
    --pred results/<视频名>_predictions.json \
    --gt ground_truth_example.json \
    --fps 30
```

---

## ❓ 常见问题速查

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| "GT中未找到视频" | 文件名不匹配 | 重命名视频文件 |
| 时间对不上 | fps设置错误 | 检查实际fps并正确设置 |
| 检测不到任何打架 | 阈值太高 | 大幅降低阈值测试 |
| 误检很多 | 阈值太低 | 提高distance和speed阈值 |
| 程序运行很慢 | 视频太大 | 使用 --vid-stride 2 跳帧 |

---

## 📞 获取帮助

详细文档:
- **您的GT格式详解**: `使用您的GT格式指南.md`
- **完整技术文档**: `README_fight_detection.md`
- **详细使用说明**: `使用说明.md`

测试工具:
- **验证GT**: `python test_gt_loading.py --gt-file <文件> --video-dir <目录>`

---

**现在开始测试吧！** 🚀

```bash
# 第一步：验证GT
python test_gt_loading.py --gt-file ground_truth_example.json --video-dir ./test_videos --fps 30

# 第二步：单视频测试
python track_with_fight_detection.py --weights <模型> --source test_videos/MyVideo_191.mp4 --gt-file ground_truth_example.json --fps 30 --show

# 第三步：可视化
python visualize_results.py --video test_videos/MyVideo_191.mp4 --pred fight_detection_results/MyVideo_191_predictions.json --gt ground_truth_example.json --fps 30
```


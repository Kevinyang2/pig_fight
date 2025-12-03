# 猪打架检测系统 - 开始使用

## 🎉 欢迎使用

这是一个完整的**基于YOLO跟踪的打架行为检测与评估系统**。系统已经完全适配您的Ground Truth格式！

---

## 📁 文件清单

### 🔧 核心程序（必需）

| 文件 | 功能 | 何时使用 |
|------|------|---------|
| `track_with_fight_detection.py` | 单视频检测与评估 | ⭐ 测试、调参 |
| `batch_evaluate.py` | 批量视频评估 | ⭐ 评估整个测试集 |
| `visualize_results.py` | 结果可视化 | ⭐ 查看检测效果 |
| `test_gt_loading.py` | GT文件验证工具 | ⭐ 第一步：验证GT |

### 📚 文档（推荐阅读）

| 文件 | 内容 | 适合 |
|------|------|------|
| `快速开始_使用您的GT.md` | ⭐⭐⭐ 5分钟快速上手 | 首次使用 |
| `使用您的GT格式指南.md` | 您的GT格式详细说明 | 了解GT格式 |
| `使用说明.md` | 完整使用流程和调优 | 深入使用 |
| `README_fight_detection.md` | 技术文档和原理 | 技术细节 |
| `QUICKSTART.md` | 通用快速入门 | 参考 |

### 📄 其他文件

| 文件 | 说明 |
|------|------|
| `ground_truth_example.json` | 您的GT文件（43个视频） |
| `track.py` | 原始跟踪脚本 |
| `create_ground_truth.py` | GT标注工具（不需要，您已有GT） |

---

## 🚀 快速开始（3步）

### 第1步：验证GT文件

```bash
python test_gt_loading.py \
    --gt-file ground_truth_example.json \
    --video-dir ./test_videos \
    --fps 30
```

**作用**: 
- ✓ 检查GT文件是否正确
- ✓ 验证视频文件名是否匹配
- ✓ 统计GT信息

---

### 第2步：单视频测试

```bash
python track_with_fight_detection.py \
    --weights runs/train/v10-APConv-AssemFormer-HSFPN-ATFLm_exp/weights/best.pt \
    --source test_videos/MyVideo_191.mp4 \
    --gt-file ground_truth_example.json \
    --fps 30 \
    --show
```

**作用**:
- ✓ 运行检测
- ✓ 与GT对比
- ✓ 输出评估指标

---

### 第3步：可视化查看

```bash
python visualize_results.py \
    --video test_videos/MyVideo_191.mp4 \
    --pred fight_detection_results/MyVideo_191_predictions.json \
    --gt ground_truth_example.json \
    --fps 30
```

**作用**:
- ✓ 看到GT和预测的对比
- ✓ 判断哪里检测对了、哪里错了
- ✓ 决定如何调整参数

---

## 📖 文档阅读顺序

### 对于首次使用者

1. **首先看**: `快速开始_使用您的GT.md`（5分钟）
   - 最快上手
   - 包含完整命令
   - 针对您的GT格式

2. **然后看**: `使用您的GT格式指南.md`（15分钟）
   - GT格式详解
   - 完整工作流程
   - 常见问题

3. **最后看**: `使用说明.md`（30分钟）
   - 参数调优策略
   - 高级技巧
   - 故障排查

### 对于需要深入了解者

4. **技术文档**: `README_fight_detection.md`
   - 检测原理
   - 评估指标
   - 代码扩展

---

## ⚙️ 系统要求

### 必需

```bash
# Python包
pip install ultralytics opencv-python numpy

# 文件
- YOLO模型权重文件
- 测试视频（文件名需匹配GT）
- GT文件（您已有）
```

### 可选但推荐

```bash
# GPU加速
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 视频信息查看
sudo apt install ffmpeg  # Linux
# 或下载 ffmpeg for Windows
```

---

## 🎯 您的GT格式说明

### 格式结构

```json
{
  "database": {
    "MyVideo_191": {
      "subset": "test",
      "annotations": [
        {"segment": ["93.1", "98.2"], "label": "fight"},
        {"segment": ["101.6", "114.0"], "label": "fight"}
      ]
    }
  }
}
```

### 重要特点

1. **时间单位**: 秒（如 `"93.1"` = 93.1秒）
2. **必须指定fps**: 系统需要fps才能转换为帧号
3. **视频名匹配**: 文件名要与GT键名一致

---

## 🔑 关键参数

### 必须设置

| 参数 | 说明 | 示例 |
|------|------|------|
| `--weights` | 模型权重文件 | `best.pt` |
| `--source` | 视频文件 | `test_videos/MyVideo_191.mp4` |
| `--gt-file` | GT文件 | `ground_truth_example.json` |
| `--fps` | 视频帧率 | `30` |

### 可调整（用于优化）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--window-size` | 30 | 滑动窗口大小（帧） |
| `--stride` | 15 | 窗口步长（帧） |
| `--distance-threshold` | 100 | 距离阈值（像素） |
| `--speed-threshold` | 50 | 速度阈值 |
| `--min-fight-duration` | 15 | 最小打架时长（帧） |
| `--conf` | 0.75 | 检测置信度 |

---

## 💡 使用技巧

### 技巧1: 先验证GT再运行

```bash
# 避免运行后才发现GT有问题
python test_gt_loading.py --gt-file ground_truth_example.json --video-dir ./test_videos --fps 30
```

### 技巧2: 单视频调参后再批量

```bash
# 在一个视频上找到最佳参数
python track_with_fight_detection.py --source test_videos/MyVideo_191.mp4 ...

# 满意后批量运行
python batch_evaluate.py --video-dir ./test_videos ...
```

### 技巧3: 使用可视化工具判断

```bash
# 不要只看数字，要看视觉效果
python visualize_results.py --video ... --pred ... --gt ...
```

### 技巧4: 检查视频实际fps

```bash
# 如果结果不对，先确认fps
ffprobe -v quiet -show_streams test_videos/MyVideo_191.mp4 | grep r_frame_rate
```

---

## 📊 评估指标解读

### 精确率 (Precision)

**含义**: 预测为打架的，真正是打架的比例

- 高精确率 → 很少误报
- 低精确率 → 误报多，需提高阈值

### 召回率 (Recall)

**含义**: 实际打架的，被检测出来的比例

- 高召回率 → 很少漏报
- 低召回率 → 漏报多，需降低阈值

### F1分数

**含义**: 精确率和召回率的平衡

- 综合评价指标
- 通常优化F1最大

---

## ⚡ 快速命令参考

### 验证GT
```bash
python test_gt_loading.py --gt-file ground_truth_example.json --video-dir ./test_videos --fps 30
```

### 单视频测试
```bash
python track_with_fight_detection.py --weights <模型> --source <视频> --gt-file ground_truth_example.json --fps 30 --show
```

### 批量评估
```bash
python batch_evaluate.py --weights <模型> --video-dir ./test_videos --gt-file ground_truth_example.json --fps 30 --output-dir results
```

### 可视化
```bash
python visualize_results.py --video <视频> --pred <预测文件> --gt ground_truth_example.json --fps 30
```

---

## 🐛 遇到问题？

### 常见错误及解决

| 错误信息 | 原因 | 解决 |
|---------|------|------|
| "GT中未找到视频" | 文件名不匹配 | 检查视频文件名 |
| "文件不存在" | 路径错误 | 检查文件路径 |
| 时间对不上 | fps错误 | 检查视频实际fps |
| 检测不到打架 | 阈值太高 | 降低阈值测试 |

### 获取详细帮助

1. 查看 `快速开始_使用您的GT.md` 的"常见问题"章节
2. 查看 `使用您的GT格式指南.md` 的"常见问题排查"章节
3. 查看 `使用说明.md` 的"问题排查"章节

---

## 📞 下一步

### 如果您是第一次使用

1. ✓ 阅读本文档（您正在看）
2. → **现在去看**: `快速开始_使用您的GT.md`
3. → 运行第一个测试
4. → 根据结果调整参数

### 如果您想深入了解

- 阅读 `使用您的GT格式指南.md`
- 阅读 `使用说明.md`
- 阅读 `README_fight_detection.md`

---

## ✅ 检查清单

使用前确认:

- [ ] 已安装依赖: `pip install ultralytics opencv-python numpy`
- [ ] 有YOLO模型权重文件
- [ ] 有测试视频文件
- [ ] 视频文件名与GT中的键名匹配
- [ ] 知道视频的实际fps

准备开始:

- [ ] 已验证GT文件: `python test_gt_loading.py ...`
- [ ] 已选择一个视频进行首次测试
- [ ] 已准备好查看可视化结果

---

**一切准备就绪！现在开始吧！** 🚀

**推荐的第一步**: 打开 `快速开始_使用您的GT.md` 并按照步骤操作。

---

## 📧 系统信息

- **版本**: 1.0
- **更新日期**: 2024-11
- **适配的GT格式**: Database格式（您的格式）
- **支持的视频格式**: MP4, AVI, MOV, MKV
- **Python版本**: 3.8+


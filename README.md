# 猪群打斗检测与事件分析

本仓库基于 Ultralytics YOLO 框架，对猪舍监控视频进行打斗行为检测与事件统计，提供从数据准备、模型训练到线上推理的一站式方案。项目聚焦于两类目标：`pig_fight` 与 `pig_normal`，并通过多模型协同与事件平滑机制，输出结构化的打斗时段结果。

## 功能亮点

- **双模型检测架构**：使用 YOLO 模型分别完成猪只检测与打斗行为识别，可自由组合不同结构（如 AIFI、AssemFormer、CBAM、ECA 等增强模块）。
- **事件融合与可视化**：`fight_pipeline.py` 聚合多模型输出并进行时间平滑，生成带时间戳的事件 CSV，同步输出带红框叠加的视频。
- **丰富的自定义模块**：`ultralytics/nn/Extramodule/` 中集成了多种注意力与特征增强模块，可在配置文件中灵活启用。
- **可复现的实验结果**：提供训练/验证脚本、可视化结果与示例配置，便于快速迁移到新的场景。

## 仓库结构速览

- `pig.yaml`：猪群打斗数据集配置（2 类，支持自定义增广）。
- `train.py` / `val.py` / `predict.py`：基于 Ultralytics YOLO CLI 的训练、验证与推理脚本扩展。
- `fight_pipeline.py`：检测与事件融合主流程脚本。
- `ultralytics/cfg/models/**`：针对 YOLOv10/v11/v12 的多种改进结构配置文件。
- `results/`、`result-v10-improved/` 等：实验可视化结果与性能曲线。

## 环境准备

1. 安装 Python 3.8+ 与 PyTorch（建议 GPU 环境）。
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 若需 TensorRT、CoreML 等导出能力，可按需放开 `requirements.txt` 中的可选依赖。

## 数据集准备

- 将标注数据按 YOLO 官方格式整理到 `pig.yaml` 中指定的路径结构：
  - `images/train`, `images/val`, `labels/train`, `labels/val`
  - 类别顺序：`0 -> pig_fight`、`1 -> pig_normal`
- 若使用自定义路径，请修改 `pig.yaml` 中的 `train`, `val`, `test` 字段。
- 数据集链接如下:（提取码可发送邮件 kevinyang0048@gmail.com 进行申请）
 https://pan.baidu.com/s/1xxO0Us1Vb_LUJCtRLB3QGg

## 模型训练

- 基于 Ultralytics CLI：
  ```bash
  # 示例：使用改进的 YOLOv10 配置进行训练
  yolo detect train \
      data=pig.yaml \
      cfg=ultralytics/cfg/models/v10/yolov10n-APConv-AssemFormer-HSFPN.yaml \
      epochs=100 imgsz=640 device=0 \
      project=runs/train name=yolov10n_pig
  ```
- 常用参数说明：
  - `cfg`：选择增强后的模型结构（仓库已提供多种版本）。
  - `imgsz`：训练与推理的输入尺寸，推荐 640。
  - `device`：GPU 编号或 `cpu`。
  - `workers`：DataLoader 线程数，取决于硬件环境。

## 模型验证与测试

- 评估指标：
  ```bash
  yolo detect val \
      data=pig.yaml \
      model=runs/train/yolov10n_pig/weights/best.pt \
      imgsz=640 batch=16 device=0
  ```
- 消融对比：仓库中 `ultralytics/cfg/models/**` 提供了多种注意力/特征融合模块，可通过切换配置验证对 mAP 与 FPS 的影响。



```bash
python fight_pipeline.py \
    --det-weights runs/train/yolov10n_pig/weights/best.pt \
    --fight-weights runs/train/yolo11_fight/weights/best.pt \
    --source data/videos/pig_demo.mp4 \
    --save-video --save-overlay --project runs/fight --name exp01
```



## 结果展示

![检测结果示例](result-v10-improved/predictions_bbox_summary.png)

- `results/mAP50(B).png`：记录主要实验在验证集上的 mAP@0.5 指标。
- `result-v10-BS*/`：不同 batch size/模型结构的预测可视化，用于对比模型稳定性。


## 致谢

- 本项目基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 开源框架进行二次开发，感谢官方团队的持续更新。
- 数据整理、模型设计与实验均面向养殖场景的真实需求，欢迎在 Issues 区交流改进建议。

---

如需更多帮助或计划投入生产环境，可通过 GitHub Issues 与我们取得联系。欢迎 Star ⭐ 与 Fork 支持项目发展！

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import argparse
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.ops import non_max_suppression, xywh2xyxy

try:
    from pytorch_grad_cam import (
        EigenCAM,
        EigenGradCAM,
        GradCAM,
        GradCAMPlusPlus,
        HiResCAM,
        LayerCAM,
        RandomCAM,
        XGradCAM,
    )
    from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image
except Exception as e:
    raise SystemExit("请先安装: pip install pytorch-grad-cam") from e


def letterbox(
    im: np.ndarray,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = round(shape[1] * r), round(shape[0] * r)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


class YoloDetTarget(torch.nn.Module):
    def __init__(self, output_type: str, conf: float, ratio: float, nc: int, class_id: int | None = None) -> None:
        super().__init__()
        self.output_type = output_type
        self.conf = conf
        self.ratio = ratio
        self.nc = nc
        self.class_id = class_id
        self.selected_indices: list[int] | None = None

    def set_selected_indices(self, idxs: list[int] | None):
        self.selected_indices = idxs if idxs is not None and len(idxs) else None

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        pred = data[0] if isinstance(data, (list, tuple)) else data
        if not torch.is_tensor(pred):
            return torch.tensor(0.0)
        if pred.ndim == 2:
            pred = pred.unsqueeze(0)
        cls_logits = pred[0, :, -self.nc :]
        boxes = pred[0, :, :4]
        if self.class_id is not None and 0 <= int(self.class_id) < self.nc:
            scores = cls_logits[:, int(self.class_id)]
        else:
            scores, _ = cls_logits.max(dim=1)
        if self.selected_indices is not None:
            idx_tensor = torch.as_tensor(self.selected_indices, device=pred.device, dtype=torch.long)
            scores = scores.index_select(0, idx_tensor)
            boxes = boxes.index_select(0, idx_tensor)
        n = scores.shape[0]
        k = max(1, min(n, int(np.ceil(n * float(self.ratio)))))
        vals, idx = torch.topk(scores, k)
        mask = vals >= self.conf
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        loss = vals[mask].sum()
        if self.output_type in ("box", "all"):
            loss = loss + boxes[idx[mask]].sum()
        return loss


def _probe_activation_hw(model, candidate_ids: list[int], imgsz: int) -> dict[int, tuple[int, int]]:
    """对候选层注册一次性 hook，前向一次记录各层 H,W。."""
    hw: dict[int, tuple[int, int]] = {}
    hooks = []

    def make_hook(idx):
        def _hook(_m, _inp, out):
            if isinstance(out, (list, tuple)):
                out = out[0]
            if torch.is_tensor(out) and out.ndim >= 4:
                _, _, h, w = out.shape[:4]
                hw[idx] = (int(h), int(w))

        return _hook

    for idx in candidate_ids:
        m = model.model[idx]
        hooks.append(m.register_forward_hook(make_hook(idx)))

    with torch.no_grad():
        dummy = torch.zeros(1, 3, imgsz, imgsz, device=next(model.parameters()).device)
        try:
            _ = model(dummy)
        except Exception:
            pass
    for h in hooks:
        h.remove()
    return hw


def auto_select_layers(model, k: int = 5, max_hw: int = 1600, imgsz: int = 640) -> list[int]:
    """优先选择尾部可训练卷积层；按激活图面积筛选不超过 max_hw 的小特征图，避免 CAM 内存爆炸。."""
    candidates = []
    for i, m in reversed(list(enumerate(model.model))):
        if isinstance(m, torch.nn.Conv2d):
            if any(p.requires_grad for p in m.parameters(recurse=False)):
                candidates.append(i)
        elif hasattr(m, "conv") and isinstance(getattr(m, "conv"), torch.nn.Conv2d):
            if any(p.requires_grad for p in m.conv.parameters(recurse=False)):
                candidates.append(i)
        if len(candidates) >= max(k * 3, k):
            break
    if not candidates:
        candidates = list(range(max(0, len(model.model) - k), len(model.model)))

    hw_map = _probe_activation_hw(model, candidates, imgsz)
    small = [i for i in candidates if i in hw_map and (hw_map[i][0] * hw_map[i][1]) <= max_hw]
    if not small:
        small = sorted(hw_map, key=lambda i: hw_map[i][0] * hw_map[i][1])[:k]
    else:
        small = sorted(small, key=lambda i: (candidates.index(i), hw_map[i][0] * hw_map[i][1]))[:k]
    return sorted(set(small))


class YoloV10Heatmap:
    def __init__(
        self,
        weights: str,
        device: str,
        method: str,
        layers: Sequence[int] | None,
        backward_type: str,
        class_id: int | None,
        conf_threshold: float,
        iou_threshold: float,
        ratio: float,
        show_box: bool,
        renormalize: bool,
        imgsz: int,
        match_iou: float = 0.3,
        topk_dets: int = 5,
    ) -> None:
        self.device = torch.device(device)
        self.model = attempt_load_weights(weights, device=self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(True)

        self.model_names = self.model.names if hasattr(self.model, "names") else {}
        self.nc = len(self.model_names) if isinstance(self.model_names, (list, tuple, dict)) else 80

        self.target = YoloDetTarget(backward_type, conf_threshold, ratio, nc=self.nc, class_id=class_id)
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.show_box = show_box
        self.renormalize = renormalize
        self.colors = np.random.uniform(0, 255, size=(self.nc, 3))
        self.match_iou = match_iou
        self.topk_dets = topk_dets

        self.layers = (
            list(layers) if layers is not None else auto_select_layers(self.model, k=3, max_hw=1600, imgsz=imgsz)
        )
        self.target_layers = [self.model.model[l] for l in self.layers]

        self.method_name = method
        self.cam = self._build_cam(method)

    def _build_cam(self, method: str):
        cam_cls = {
            "GradCAM": GradCAM,
            "GradCAMPlusPlus": GradCAMPlusPlus,
            "XGradCAM": XGradCAM,
            "EigenCAM": EigenCAM,
            "HiResCAM": HiResCAM,
            "LayerCAM": LayerCAM,
            "RandomCAM": RandomCAM,
            "EigenGradCAM": EigenGradCAM,
        }[method]
        return cam_cls(self.model, self.target_layers, use_cuda=self.device.type == "cuda")

    def _post_process(self, pred: torch.Tensor) -> torch.Tensor:
        result = non_max_suppression(pred, conf_thres=self.conf_threshold, iou_thres=self.iou_threshold)[0]
        return result

    def _draw_det(self, box, color, name, img):
        x1, y1, x2, y2 = list(map(int, list(box)))
        cv2.rectangle(img, (x1, y1), (x2, y2), tuple(int(x) for x in color), 2)
        cv2.putText(
            img,
            str(name),
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            tuple(int(x) for x in color),
            2,
            lineType=cv2.LINE_AA,
        )
        return img

    def _renorm_cam_in_boxes(self, boxes, image_float_np, grayscale_cam):
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        return show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)

    def process_one(self, img_path: str, save_path: str) -> None:
        img0 = cv2.imread(img_path)
        if img0 is None:
            return
        img, _, _ = letterbox(img0, new_shape=(self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img_float, (2, 0, 1))).unsqueeze(0).to(self.device)

        # 先推理以选择指定类别的检测，约束 CAM 目标
        with torch.no_grad():
            pred_raw = self.model(tensor)[0]
        det = self._post_process(pred_raw)
        selected_indices: list[int] | None = None
        if det is not None and len(det):
            det_for_match = det
            if self.target.class_id is not None and det.shape[1] >= 6:
                det_for_match = det_for_match[det_for_match[:, 5] == float(int(self.target.class_id))]
            if len(det_for_match) > self.topk_dets:
                order = torch.argsort(det_for_match[:, 4], descending=True)[: self.topk_dets]
                det_for_match = det_for_match.index_select(0, order)
            if len(det_for_match):
                pr = pred_raw[0] if pred_raw.ndim == 3 else pred_raw
                pred_boxes_xyxy = xywh2xyxy(pr[:, :4]) if pr.ndim == 2 and pr.shape[1] >= 4 else None
                if pred_boxes_xyxy is not None:
                    selected_indices = []
                    pb = pred_boxes_xyxy
                    pb_np = pb.cpu().numpy()
                    for b in det_for_match[:, :4]:
                        b = b.cpu().numpy()
                        x1 = np.maximum(pb_np[:, 0], b[0])
                        y1 = np.maximum(pb_np[:, 1], b[1])
                        x2 = np.minimum(pb_np[:, 2], b[2])
                        y2 = np.minimum(pb_np[:, 3], b[3])
                        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
                        area_p = (pb_np[:, 2] - pb_np[:, 0]) * (pb_np[:, 3] - pb_np[:, 1])
                        area_b = (b[2] - b[0]) * (b[3] - b[1])
                        union = area_p + area_b - inter + 1e-6
                        ious = inter / union
                        mi = int(np.argmax(ious))
                        if ious[mi] >= self.match_iou:
                            selected_indices.append(mi)
                    if not selected_indices:
                        selected_indices = None
        self.target.set_selected_indices(selected_indices)

        # 生成 CAM：失败则回退 GradCAM -> 更小层 -> EigenCAM
        try:
            grayscale_cam = self.cam(tensor, targets=[self.target])[0]
        except Exception:
            if self.method_name != "GradCAM":
                self.method_name = "GradCAM"
                self.cam = self._build_cam(self.method_name)
                try:
                    grayscale_cam = self.cam(tensor, targets=[self.target])[0]
                except Exception:
                    pass
            if "grayscale_cam" not in locals():
                self.layers = auto_select_layers(self.model, k=2, max_hw=800, imgsz=self.imgsz)
                self.target_layers = [self.model.model[l] for l in self.layers]
                self.cam = self._build_cam(self.method_name)
                try:
                    grayscale_cam = self.cam(tensor, targets=[self.target])[0]
                except Exception:
                    pass
            if "grayscale_cam" not in locals():
                self.method_name = "EigenCAM"
                self.cam = self._build_cam(self.method_name)
                grayscale_cam = self.cam(tensor, targets=[self.target])[0]

        cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

        with torch.no_grad():
            pred = self.model(tensor)[0]
        det = self._post_process(pred)

        if self.renormalize and det is not None and len(det):
            boxes = det[:, :4].cpu().numpy().astype(np.int32)
            cam_image = self._renorm_cam_in_boxes(boxes, img_float, grayscale_cam)

        if self.show_box and det is not None and len(det):
            for d in det:
                d = d.cpu().numpy()
                cls_idx = int(d[5]) if d.shape[0] >= 6 else int(np.argmax(d[4:]))
                score = float(d[4]) if d.shape[0] >= 6 else float(np.max(d[4:]))
                name = (
                    f"{self.model_names.get(cls_idx, cls_idx)} {score:.2f}"
                    if isinstance(self.model_names, dict)
                    else f"{cls_idx} {score:.2f}"
                )
                cam_image = self._draw_det(d[:4], self.colors[cls_idx], name, cam_image)

        Image.fromarray(cam_image).save(save_path)

    def __call__(self, source: str, save_dir: str) -> None:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        p = Path(source)
        if p.is_dir():
            for name in sorted(os.listdir(str(p))):
                ip = str(p / name)
                if not os.path.isfile(ip):
                    continue
                sp = str(Path(save_dir) / name)
                self.process_one(ip, sp)
        else:
            sp = str(Path(save_dir) / "result.png")
            self.process_one(str(p), sp)


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv10 Grad-CAM 热力图生成")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/train/v10-BS_exp/weights/best.pt",
        help="权重文件路径(.pt)",
    )
    parser.add_argument("--source", type=str, required=True, help="输入图像或文件夹路径")
    parser.add_argument("--save-dir", type=str, default="result-heatmap", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备 cuda:0/cpu")
    parser.add_argument(
        "--method",
        type=str,
        default="XGradCAM",
        choices=[
            "GradCAM",
            "GradCAMPlusPlus",
            "XGradCAM",
            "EigenCAM",
            "HiResCAM",
            "LayerCAM",
            "RandomCAM",
            "EigenGradCAM",
        ],
        help="CAM 方法",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="auto",
        help="用于 CAM 的层索引(逗号分隔)，或 'auto' 自动选择",
    )
    parser.add_argument("--backward-type", type=str, default="class", choices=["class", "box", "all"], help="目标类型")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.65, help="IOU 阈值")
    parser.add_argument("--ratio", type=float, default=0.05, help="top-k 比例 (0-1)")
    parser.add_argument("--show-box", action="store_true", help="是否叠加检测框")
    parser.add_argument("--renormalize", action="store_true", help="检测框内重归一化热力图")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸(方形)")
    parser.add_argument("--class-id", type=int, default=None, help="指定可视化的类别ID（例如猪的类别索引）")
    parser.add_argument("--match-iou", type=float, default=0.3, help="将 CAM 约束到与检测框 IoU>=此阈值的预测行")
    parser.add_argument("--topk-dets", type=int, default=5, help="用于构造 CAM 的前K个高分检测框")
    args = parser.parse_args()
    if args.layers.strip().lower() == "auto":
        layers = None
    else:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    return args, layers


def main():
    args, layers = parse_args()
    cam = YoloV10Heatmap(
        weights=args.weights,
        device=args.device,
        method=args.method,
        layers=layers,
        backward_type=args.backward_type,
        class_id=args.class_id,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        ratio=args.ratio,
        show_box=args.show_box,
        renormalize=args.renormalize,
        imgsz=args.imgsz,
        match_iou=args.match_iou,
        topk_dets=args.topk_dets,
    )
    cam(args.source, args.save_dir)


if __name__ == "__main__":
    main()

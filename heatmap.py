import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
import math
import os
import shutil

import cv2
import numpy as np
import torch

np.random.seed(0)
from PIL import Image
from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image

from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.ops import non_max_suppression


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = round(shape[1] * r), round(shape[0] * r)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


## 使用 pytorch-grad-cam 自带的 ActivationsAndGradients，删除自定义覆盖类


class yolov8_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio, nc: int) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio
        self.nc = nc

    def forward(self, data):
        # data can be a Tensor (B, A, no) or a list/tuple with first element as Tensor
        pred = data[0] if isinstance(data, (list, tuple)) else data
        if not torch.is_tensor(pred):
            return torch.tensor(0.0)
        # ensure 3D shape
        if pred.ndim == 2:
            pred = pred.unsqueeze(0)
        # classification logits are last nc channels
        cls_logits = pred[0, :, -self.nc :]
        # optional: boxes first 4 channels (not used when ouput_type='class')
        boxes = pred[0, :, :4]
        scores, _ = cls_logits.max(dim=1)
        # select top-k by ratio with confidence threshold
        n = cls_logits.shape[0]
        k = max(1, min(n, math.ceil(n * float(self.ratio))))
        vals, idx = torch.topk(scores, k)
        mask = vals >= self.conf
        selected_scores = vals[mask]
        if selected_scores.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        loss = selected_scores.sum()
        if self.ouput_type in ("box", "all"):
            sel_boxes = boxes[idx[mask]]
            loss = loss + sel_boxes.sum()
        return loss


class yolov10_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_box, renormalize):
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt["model"].names
        model = attempt_load_weights(weight, device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()

        # 构造目标函数，传入类别数以从预测张量尾部切分类别 logits
        target = yolov8_target(backward_type, conf_threshold, ratio, nc=len(model_names))
        target_layers = [model.model[l] for l in layer]
        method = eval(method)(model, target_layers, use_cuda=device.type == "cuda")

        colors = np.random.uniform(0, 255, size=(len(model_names), 3))
        self.__dict__.update(locals())

    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.65)[0]
        return result

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(
            img,
            str(name),
            (xmin, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            tuple(int(x) for x in color),
            2,
            lineType=cv2.LINE_AA,
        )
        return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1] inside every bounding boxes, and zero outside of the bounding
        boxes.
        """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized

    def process(self, img_path, save_path):
        # img process
        img = cv2.imread(img_path)
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        try:
            grayscale_cam = self.method(tensor, [self.target])
        except AttributeError:
            return

        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

        pred = self.model(tensor)[0]
        pred = self.post_process(pred)
        if self.renormalize:
            cam_image = self.renormalize_cam_in_bounding_boxes(
                pred[:, :4].cpu().detach().numpy().astype(np.int32), img, grayscale_cam
            )
        if self.show_box:
            for data in pred:
                data = data.cpu().detach().numpy()
                cam_image = self.draw_detections(
                    data[:4],
                    self.colors[int(data[4:].argmax())],
                    f"{self.model_names[int(data[4:].argmax())]} {float(data[4:].max()):.2f}",
                    cam_image,
                )

        cam_image = Image.fromarray(cam_image)
        cam_image.save(save_path)

    def __call__(self, img_path, save_path):
        # remove dir if exist
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        # make dir if not exist
        os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(img_path):
            for img_path_ in os.listdir(img_path):
                self.process(f"{img_path}/{img_path_}", f"{save_path}/{img_path_}")
        else:
            self.process(img_path, f"{save_path}/result.png")


def get_params():
    params = {
        "weight": "runs/train/v10-BS_exp/weights/best.pt",  # 只需要指定权重即可（或自定义 best.pt）
        "device": "cuda:0",
        "method": "XGradCAM",
        # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
        "layer": [10, 12, 14, 16, 18],
        "backward_type": "class",  # class, box, all
        "conf_threshold": 0.2,  # 0.2
        "ratio": 0.02,  # 0.02-0.1
        "show_box": False,
        "renormalize": True,
    }
    return params


if __name__ == "__main__":
    model = yolov10_heatmap(**get_params())
    model("F:\生猪争斗行为视频数据\\test_frames\\video_20201218_144915_264\\000003.jpg", "result-heatmap")

import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, cv2, os, shutil, copy
import numpy as np

np.random.seed(0)
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
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
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (top, bottom, left, right)


class ActivationsAndGradients:
    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(target_layer.register_forward_hook(self.save_activation))
            self.handles.append(target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, 'requires_grad') or not output.requires_grad:
            return

        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):
        # 仅支持 detect（YOLOv10）
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted_scores, indices = torch.sort(logits_.max(1)[0], descending=True)
        logits_sorted = torch.transpose(logits_[0], 0, 1)[indices[0]]
        boxes_sorted = torch.transpose(boxes_[0], 0, 1)[indices[0]]
        return logits_sorted, boxes_sorted

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        post_result, pre_post_boxes = self.post_process(model_output[0])
        return [[post_result, pre_post_boxes]]

    def release(self):
        for handle in self.handles:
            handle.remove()


class yolo_detect_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio


    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        topn = max(1, int(post_result.size(0) * self.ratio))
        for i in range(topn):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type in ('class', 'all'):
                if self.class_id is not None and 0 <= int(self.class_id) < post_result.shape[1]:
                    result.append(post_result[i, int(self.class_id)])
                else:
                    result.append(post_result[i].max())
            if self.ouput_type in ('box', 'all'):
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result) if len(result) else torch.tensor(0.0, device=pre_post_boxes.device)


class yolov10_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_result, renormalize,
                 img_size):
        device = torch.device(device)
        model_yolo = YOLO(weight)
        model_names = model_yolo.names
        print(f'model class info:{model_names}')
        model = copy.deepcopy(model_yolo.model)
        model.to(device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()

        target = yolo_detect_target(backward_type, conf_threshold, ratio)
        target_layers = [model.model[l] for l in layer]
        cam_method = eval(method)(model, target_layers)
        cam_method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)

        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)
        self.__dict__.update(locals())

    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.65)[0]
        return result

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized

    def process(self, img_path, save_path):
        try:
            img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            print(f"Warning... {img_path} read failure.")
            return
        img, _, (top, bottom, left, right) = letterbox(img, new_shape=(self.img_size, self.img_size), auto=True)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        try:
            grayscale_cam = self.cam_method(tensor, [self.target])
        except AttributeError:
            print("Warning... CAM forward failure.")
            return
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

        pred = self.model_yolo.predict(tensor, conf=self.conf_threshold, iou=0.7)[0]
        if self.renormalize:
            cam_image = self.renormalize_cam_in_bounding_boxes(pred.boxes.xyxy.cpu().detach().numpy().astype(np.int32),
                                                               img, grayscale_cam)
        if self.show_result:
            cam_image = pred.plot(img=cam_image, conf=True, font_size=None, line_width=None, labels=False)

        cam_image = cam_image[top:cam_image.shape[0] - bottom, left:cam_image.shape[1] - right]
        cam_image = Image.fromarray(cam_image)
        cam_image.save(save_path)

    def __call__(self, img_path, save_path):
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)
        if os.path.isdir(img_path):
            for img_path_ in os.listdir(img_path):
                self.process(f'{img_path}/{img_path_}', f'{save_path}/{img_path_}')
        else:
            self.process(img_path, f'{save_path}/result.png')


def get_params():
    return {
        'weight': 'runs/train/v10-APConv+ATFLm_exp/weights/best.pt',
        'device': 'cuda:0',
        'method': 'EigenCAM',
        'layer': [16, 19, 22],
        'backward_type': 'class',
        'conf_threshold': 0.25,
        'ratio': 0.02,
        'show_result': True,
        'renormalize': True,
        'img_size': 640,
    }


if __name__ == '__main__':
    params = get_params()
    model = yolov10_heatmap(**params)
    # 示例：单张图片
    model(r"D:\googleDownload\pig_fight_DEC\valid\images\000001_jpg.rf.2dcec7980067e47995dd7abfe1ce04ef.jpg", 'result-heatmap-AP-ATFL')
    # 或者：目录
    # model(r"D:\\path\\to\\images_dir", 'result-heatmap')


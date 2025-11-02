import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Union
from ultralytics.nn.modules.head import Detect

class DetectDAD(Detect):
    """
    基于 YOLO Detect 的难度感知头：在分类 logits 上加入按类的动态偏置 beta(d_hat)。
    继承 Detect，保持训练/推理与 NMS、解码完全兼容。
    """

    def __init__(self, nc: int = 80, ch: Tuple = (), D: int = 64, d_hat=None):
        super().__init__(nc, ch)
        self.D = D
        self.mlp = nn.Sequential(
            nn.Linear(1, D), nn.ReLU(inplace=True), nn.Linear(D, D), nn.ReLU(inplace=True)
        )
        self.proj_beta = nn.Linear(D, 1, bias=False)
        # d_hat buffer
        if d_hat is None:
            d_hat_t = torch.zeros(nc, dtype=torch.float)
        else:
            d_hat_t = torch.as_tensor(d_hat, dtype=torch.float).view(-1)
            if d_hat_t.numel() != nc:
                raise ValueError(f"d_hat length {d_hat_t.numel()} != num_cls {nc}")
        self.register_buffer("d_hat", d_hat_t)

    def _beta(self) -> torch.Tensor:
        e = self.mlp(self.d_hat.view(self.nc, 1))          # [C, D]
        beta = self.proj_beta(e).view(1, self.nc, 1, 1)    # [1,C,1,1]
        return beta

    def forward(self, x: List[torch.Tensor]) -> Union[List[torch.Tensor], Tuple]:
        if self.end2end:
            return self.forward_end2end(x)

        beta = self._beta()
        for i in range(self.nl):
            # reg 与 Detect 一致
            reg = self.cv2[i](x[i])
            # 分类分支输出 + beta
            cls = self.cv3[i](x[i]) + beta
            x[i] = torch.cat((reg, cls), 1)

        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    @torch.no_grad()
    def update_d_hat(self, d_hat):
        device = self.d_hat.device
        dh = torch.as_tensor(d_hat, dtype=self.d_hat.dtype, device=device).view(self.nc)
        self.d_hat.data.copy_(dh)

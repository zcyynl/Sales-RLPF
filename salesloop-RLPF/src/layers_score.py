"""
layers_score.py

layers.py 的轻量版，仅供 score_for_reward.py 使用。
去掉 deepspeed 依赖（pydantic v2 环境下 deepspeed 导入报错），
add_lora 逻辑与 layers.py 完全一致。
"""

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        if x.dtype != self.A.dtype:
            A = self.A.to(x.dtype)
            B = self.B.to(x.dtype)
        else:
            A, B = self.A, self.B
        return self.alpha * (x @ A @ B)


class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha, pissa=False):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def add_lora(model, r, alpha, pissa):
    for layer in model.model.layers:
        layer.self_attn.q_proj = LinearWithLoRA(layer.self_attn.q_proj, r, alpha, pissa)
        layer.self_attn.k_proj = LinearWithLoRA(layer.self_attn.k_proj, r, alpha, pissa)
        layer.self_attn.v_proj = LinearWithLoRA(layer.self_attn.v_proj, r, alpha, pissa)
        layer.self_attn.o_proj = LinearWithLoRA(layer.self_attn.o_proj, r, alpha, pissa)
    print("******************HERE (add_lora done)")

"""
ann_model.py
============
논문 §3 의 feedforward MLP. yaml 의 model 섹션을 받아 동적 생성.

구조:
    input → [Linear → activation → (dropout)] × N → Linear → (output_activation)

기본값:
    hidden_layers : [64, 64, 64]
    activation    : tanh
    dropout       : 0.0
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────
# Activation 매핑
# ──────────────────────────────────────────────────────────────
_ACTIVATIONS = {
    'tanh' : nn.Tanh,
    'relu' : nn.ReLU,
    'gelu' : nn.GELU,
    'silu' : nn.SiLU,
    'leaky_relu': nn.LeakyReLU,
}


def _make_activation(name):
    name = (name or 'tanh').lower()
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"unsupported activation: {name} "
            f"(지원: {list(_ACTIVATIONS.keys())})")
    return _ACTIVATIONS[name]()


# ──────────────────────────────────────────────────────────────
# MLP
# ──────────────────────────────────────────────────────────────
class MLP(nn.Module):
    """
    fully-connected feedforward network.

    Parameters
    ----------
    input_dim         : int
    output_dim        : int
    hidden_layers     : list[int]  각 hidden layer 의 뉴런 수
    activation        : str        hidden activation 이름
    dropout           : float      각 hidden 뒤 dropout 확률
    output_activation : str | None 출력 activation (None = 선형)
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layers,
        activation='tanh',
        dropout=0.0,
        output_activation=None,
    ):
        super().__init__()

        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(_make_activation(activation))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = h
        # 출력 layer
        layers.append(nn.Linear(prev, output_dim))
        if output_activation:
            layers.append(_make_activation(output_activation))

        self.net = nn.Sequential(*layers)

        # 메타데이터 보관 (config 재현용)
        self.config = {
            'input_dim'        : int(input_dim),
            'output_dim'       : int(output_dim),
            'hidden_layers'    : list(map(int, hidden_layers)),
            'activation'       : str(activation),
            'dropout'          : float(dropout),
            'output_activation': output_activation,
        }

    def forward(self, x):
        return self.net(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────
# YAML config → 모델 인스턴스
# ──────────────────────────────────────────────────────────────
def build_model_from_config(model_cfg):
    """network.yaml 의 model: 섹션 dict → MLP 인스턴스."""
    mtype = model_cfg.get('type', 'mlp').lower()
    if mtype != 'mlp':
        raise ValueError(f"unsupported model type: {mtype} (지원: 'mlp')")
    return MLP(
        input_dim         = int(model_cfg['input_dim']),
        output_dim        = int(model_cfg['output_dim']),
        hidden_layers     = list(model_cfg['hidden_layers']),
        activation        = str(model_cfg.get('activation', 'tanh')),
        dropout           = float(model_cfg.get('dropout', 0.0)),
        output_activation = model_cfg.get('output_activation', None),
    )
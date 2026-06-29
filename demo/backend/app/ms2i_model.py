import os
import glob
import sys
from datetime import datetime
import random
import cv2
import numbers
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union
from dataclasses import dataclass, asdict
import math
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MONARCH_ATTENTION_ROOT = _PROJECT_ROOT / "monarch-attention"
print(f"MONARCH_ATTENTION_ROOT: {_MONARCH_ATTENTION_ROOT}")
if _MONARCH_ATTENTION_ROOT.exists() and str(_MONARCH_ATTENTION_ROOT) not in sys.path:
    print("Adding Monarch Attention to sys.path for imports.")
    sys.path.insert(0, str(_MONARCH_ATTENTION_ROOT))

_MONARCH_IMPORT_SOURCE = None
try:
    from ma.monarch_attention import MonarchAttention  # pyright: ignore[reportMissingImports]
    _MONARCH_IMPORT_SOURCE = "ma.monarch_attention"
except ImportError:
    try:
        from monarch_attn import MonarchAttention  # pyright: ignore[reportMissingImports]
        _MONARCH_IMPORT_SOURCE = "monarch_attn"
    except ImportError:
        raise ImportError(
            "Unable to import MonarchAttention from the bundled monarch-attention repo. "
            "Verify that monarch-attention/ma/__init__.py exists and that the repo root is on sys.path."
        )

# Log which MonarchAttention implementation was used
import logging
_logger = logging.getLogger(__name__)
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    _logger.addHandler(_handler)
_logger.info("MonarchAttention import source: %s", _MONARCH_IMPORT_SOURCE)


COLOR_LABELS = (
    "Black",
    "White",
    "Gray",
    "Red",
    "Orange",
    "Yellow",
    "Green",
    "Blue",
    "Purple",
    "Pink",
    "Brown",
)
COLOR_TO_INDEX = {label.lower(): idx for idx, label in enumerate(COLOR_LABELS)}


def rearrange(x, pattern, **axes_lengths):
    if pattern == 'b c h w -> b (h w) c':
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], x.shape[2] * x.shape[3], x.shape[1])
    if pattern == 'b (h w) c -> b c h w':
        h = axes_lengths['h']
        w = axes_lengths['w']
        return x.reshape(x.shape[0], h, w, x.shape[2]).permute(0, 3, 1, 2)
    if pattern == 'b (head c) h w -> b head c (h w)':
        head = axes_lengths['head']
        b, channels, h, w = x.shape
        c = channels // head
        return x.reshape(b, head, c, h * w)
    if pattern == 'b head c (h w) -> b (head c) h w':
        head = axes_lengths['head']
        h = axes_lengths['h']
        w = axes_lengths['w']
        b, _, c, _ = x.shape
        return x.reshape(b, head * c, h, w)
    raise NotImplementedError(f"Unsupported rearrange pattern: {pattern}")


def count_params(model):
    """ Count model trainable parameters """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')
print(f'Number of available cpu: {os.cpu_count()}')

# Standard library imports
import os
import json
import time
import random
import math
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional

# Third-party imports
import numpy as np
import torch
try:
    import albumentations as A  # pyright: ignore[reportMissingImports]
except ImportError:
    A = None
import cv2
from PIL import Image, ImageOps
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# PyTorch data utilities
from torch.utils.data import Dataset, DataLoader

class UpSample(nn.Module):
    """ UpSampling block using PixelShuffle """
    def __init__(self, filters=64):
        super().__init__()
        self.conv = nn.Conv2d(filters, filters * 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

## DownSampling block
class DownSample(nn.Module):
    """ DownSampling block using PixelUnshuffle """
    def __init__(self, filters=64):
        super().__init__()
        self.conv = nn.Conv2d(filters, filters // 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2)

    def forward(self, x):
        """ SHAPE (B, C, H, W) -> SHAPE (B, C/4, H/2, W/2) """
        x = self.conv(x)
        x = self.pixel_unshuffle(x)
        return x

class ConvDown(nn.Module):
    """ DownSampling block using strided convolution """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvUp(nn.Module):
    """ UpSampling block using Upsample + convolution """
    def __init__(self, in_channels, out_channels, out_shape=None, scale_factor=None):
        super().__init__()
        assert (out_shape is not None) ^ (scale_factor is not None), "Either out_shape or scale_factor must be provided, but not both."
        if out_shape:
            self.out_shape = out_shape
            self.upsample = nn.Upsample(out_shape, mode='bilinear', align_corners=False)
        else:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

def shape_estimation(h, w, kernel_size=3, stride=1, padding=1):
    """ Estimate the output shape after a convolutional layer. """
    out_h = (h + 2*padding - kernel_size) // stride + 1
    out_w = (w + 2*padding - kernel_size) // stride + 1
    return out_h, out_w

# Basic blocks
class DConvBlock(nn.Module):
    """ Custom Depthwise Convolution Block """
    def __init__(self, inshape, dim=64, expansion_factor=1.0, bias=False):
        super().__init__()
        hidden_features = int(dim*expansion_factor)
        self.conv = nn.Conv2d(inshape, hidden_features, kernel_size=1, bias=bias)
        self.depthwise = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.depthwise(x)
        return x

# Custom LayerNormalization
class BiasFree_LayerNorm(nn.Module):
    """ Bias-Free Layer Normalization """
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        x = x.contiguous() 
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    """ With-Bias Layer Normalization """
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        x = x.contiguous() 
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    
class LayerNorm(nn.Module):
    """ Layer Normalization supporting two types: BiasFree and WithBias """
    def __init__(self, dim, LayerNorm_type, out_4d=True):
        super().__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
        self.out_4d = out_4d

    def to_3d(self, x):
        # Convert (B, C, H, W) to (B, H*W, C)
        if len(x.shape) == 3:
            return x
        elif len(x.shape) == 4:
            return rearrange(x, 'b c h w -> b (h w) c')
        else:
            raise ValueError("Input must be a 3D or 4D tensor")
    
    def to_4d(self, x, h, w):
        # Convert (B, H*W, C) to (B, C, H, W)
        if len(x.shape) == 4:
            return x
        elif len(x.shape) == 3:
            return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        else:
            raise ValueError("Input must be a 3D or 4D tensor")

    def forward(self, x):
        if self.out_4d:
            h, w = x.shape[-2:]
            return self.to_4d(self.body(self.to_3d(x)), h, w)
        else:
            return self.body(x)

class RepConv3(nn.Module):
    def __init__(self, in_channels, out_channels, groups, deploy=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.deploy = deploy
        self.reparam = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)
        if not deploy:
            self.conv_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)
            self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups)
            self.conv_1x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), groups=groups)
            self.conv_3x1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), groups=groups)
            self.conv_1x1_branch = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=groups, bias=False)
            self.conv_3x3_branch = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups, bias=False)
        else:
            self._delete_branches()

    def _delete_branches(self):
        for name in ['conv_3x3','conv_1x1','conv_1x3','conv_3x1', 'conv_1x1_branch', 'conv_3x3_branch']:
            if hasattr(self, name):
                delattr(self, name)

    def fuse(self, delete_branches=True):
        if self.deploy:
            return
        # Extract weights and biases
        conv_3x3_w, conv_3x3_b = self.conv_3x3.weight, self.conv_3x3.bias
        conv_1x1_w, conv_1x1_b = self.conv_1x1.weight, self.conv_1x1.bias
        conv_1x3_w, conv_1x3_b = self.conv_1x3.weight, self.conv_1x3.bias
        conv_3x1_w, conv_3x1_b = self.conv_3x1.weight, self.conv_3x1.bias
        conv_1x1_branch_w, conv_3x3_branch_w = self.conv_1x1_branch.weight, self.conv_3x3_branch.weight
        # Pad the smaller kernels to 3x3
        conv_1x1_w_pad = F.pad(conv_1x1_w, [1, 1, 1, 1])
        conv_1x3_w_pad = F.pad(conv_1x3_w, [0, 0, 1, 1])
        conv_3x1_w_pad = F.pad(conv_3x1_w, [1, 1, 0, 0])
        if self.groups == 1:
            conv_1x1_3x3_w_pad = F.conv2d(conv_3x3_branch_w, conv_1x1_branch_w.permute(1, 0, 2, 3))
        else:
            w_slices = []
            conv_1x1_branch_w_T = conv_1x1_branch_w.permute(1, 0, 2, 3)
            in_channels_per_group = self.in_channels // self.groups
            out_channels_per_group = self.out_channels // self.groups
            for g in range(self.groups):
                # Slice the transposed 1x1 weights for this group's channels
                conv_1x1_branch_w_T_slice = conv_1x1_branch_w_T[:, g*in_channels_per_group:(g+1)*in_channels_per_group, :, :]
                # Slice the 3x3 weights for this group's output channels
                conv_3x3_branch_w_slice = conv_3x3_branch_w[g*out_channels_per_group:(g+1)*out_channels_per_group, :, :, :]
                w_slices.append(F.conv2d(conv_3x3_branch_w_slice, conv_1x1_branch_w_T_slice))
            conv_1x1_3x3_w_pad = torch.cat(w_slices, dim=0)
        # Fuse weights and biases
        conv_w = conv_3x3_w + conv_1x1_w_pad + conv_1x3_w_pad + conv_3x1_w_pad + conv_1x1_3x3_w_pad
        if conv_3x3_b is None:
            conv_3x3_b = torch.zeros(self.out_channels, device=conv_w.device)
        conv_b = conv_3x3_b + conv_1x1_b + conv_1x3_b + conv_3x1_b
        self.reparam.weight.data.copy_(conv_w)
        self.reparam.bias.data.copy_(conv_b)
        # Delete the original branches
        if delete_branches:
            self._delete_branches()
        # Set deploy flag
        self.deploy = True

    def forward(self, x):
        if self.deploy:
            return self.reparam(x)
        else:
            return self.conv_3x3(x) + self.conv_1x1(x) + self.conv_1x3(x) + self.conv_3x1(x) + self.conv_3x3_branch(self.conv_1x1_branch(x))

# # Test repconv
# conv = RepConv3(16, 32, groups=16, deploy=False)
# x = torch.randn(1, 16, 64, 64)
# out1 = conv(x)
# print(f"Before fusion: {count_params(conv)} parameters")
# conv.fuse()
# out2 = conv(x)
# print(f"After fusion: {count_params(conv)} parameters")
# print(torch.allclose(out1, out2, atol=1e-5))

class RepConv5(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, deploy=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.deploy = deploy
        self.reparam = nn.Conv2d(in_channels, out_channels, 5, padding=2, groups=groups)

        if not deploy:
            # Main branches
            self.conv_5x5 = nn.Conv2d(in_channels, out_channels, 5, padding=2, groups=groups)
            self.conv_3x3 = nn.Conv2d(in_channels, out_channels, 3, padding=1, groups=groups)
            self.conv_1x1 = nn.Conv2d(in_channels, out_channels, 1, groups=groups)
            # Asymmetric branches
            self.conv_1x5 = nn.Conv2d(in_channels, out_channels, (1,5), padding=(0,2), groups=groups)
            self.conv_5x1 = nn.Conv2d(in_channels, out_channels, (5,1), padding=(2,0), groups=groups)
            self.conv_1x3 = nn.Conv2d(in_channels, out_channels, (1,3), padding=(0,1), groups=groups)
            self.conv_3x1 = nn.Conv2d(in_channels, out_channels, (3,1), padding=(1,0), groups=groups)
            self.conv_3x5 = nn.Conv2d(in_channels, out_channels, (3,5), padding=(1,2), groups=groups)
            self.conv_5x3 = nn.Conv2d(in_channels, out_channels, (5,3), padding=(2,1), groups=groups)
            # Sequential branch
            self.conv_1x1_branch = nn.Conv2d(in_channels, in_channels, 1, groups=groups, bias=False)
            self.conv_5x5_branch = nn.Conv2d(in_channels, out_channels, 5, padding=2, groups=groups, bias=False)
        else:
            self._delete_branches()

    def _delete_branches(self):
        for name in [
            'conv_5x5','conv_3x3','conv_1x1',
            'conv_1x5','conv_5x1',
            'conv_1x3','conv_3x1',
            'conv_3x5','conv_5x3',
            'conv_1x1_branch','conv_5x5_branch'
        ]:
            if hasattr(self, name):
                delattr(self, name)

    def _pad_to_5x5(self, w):
        _, _, h, w_k = w.shape
        pad_h = 5 - h
        pad_w = 5 - w_k
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return F.pad(w, [pad_left, pad_right, pad_top, pad_bottom])

    def fuse(self, delete_branches=True):
        if self.deploy:
            return
        def get_wb(conv):
            return conv.weight, conv.bias if conv.bias is not None else torch.zeros(self.out_channels, device=conv.weight.device)
        W = 0
        B = 0
        # Helper to accumulate
        def add_branch(w, b):
            nonlocal W, B
            W = W + w
            B = B + b
        # Main kernels
        w, b = get_wb(self.conv_5x5)
        add_branch(w, b)
        w, b = get_wb(self.conv_3x3)
        add_branch(self._pad_to_5x5(w), b)
        w, b = get_wb(self.conv_1x1)
        add_branch(self._pad_to_5x5(w), b)
        # Asymmetric
        w, b = get_wb(self.conv_1x5)
        add_branch(self._pad_to_5x5(w), b)
        w, b = get_wb(self.conv_5x1)
        add_branch(self._pad_to_5x5(w), b)
        w, b = get_wb(self.conv_1x3)
        add_branch(self._pad_to_5x5(w), b)
        w, b = get_wb(self.conv_3x1)
        add_branch(self._pad_to_5x5(w), b)
        w, b = get_wb(self.conv_3x5)
        add_branch(self._pad_to_5x5(w), b)
        w, b = get_wb(self.conv_5x3)
        add_branch(self._pad_to_5x5(w), b)
        # Sequential 1x1 -> 5x5
        w1 = self.conv_1x1_branch.weight
        w2 = self.conv_5x5_branch.weight
        if self.groups == 1:
            w_seq = F.conv2d(w2, w1.permute(1,0,2,3))
        else:
            w_slices = []
            w1_T = w1.permute(1,0,2,3)
            icpg = self.in_channels // self.groups
            ocpg = self.out_channels // self.groups
            for g in range(self.groups):
                w1_slice = w1_T[:, g*icpg:(g+1)*icpg]
                w2_slice = w2[g*ocpg:(g+1)*ocpg]
                w_slices.append(F.conv2d(w2_slice, w1_slice))
            w_seq = torch.cat(w_slices, dim=0)
        add_branch(w_seq, torch.zeros(self.out_channels, device=w_seq.device))
        self.reparam.weight.data.copy_(W)
        self.reparam.bias.data.copy_(B)
        if delete_branches:
            self._delete_branches()
        self.deploy = True

    def forward(self, x):
        if self.deploy:
            return self.reparam(x)
        return (
            self.conv_5x5(x)
            + self.conv_3x3(x)
            + self.conv_1x1(x)
            + self.conv_1x5(x)
            + self.conv_5x1(x)
            + self.conv_1x3(x)
            + self.conv_3x1(x)
            + self.conv_3x5(x)
            + self.conv_5x3(x)
            + self.conv_5x5_branch(self.conv_1x1_branch(x))
        )

# # Test RepConv5
# conv = RepConv5(16, 32, groups=16, deploy=False)
# x = torch.randn(1, 16, 64, 64)
# out1 = conv(x)
# print(f"Before fusion: {count_params(conv)} parameters")
# conv.fuse()
# out2 = conv(x)
# print(f"After fusion: {count_params(conv)} parameters")
# print(torch.allclose(out1, out2, atol=1e-5))

class RepConv7(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, deploy=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.deploy = deploy

        self.reparam = nn.Conv2d(in_channels, out_channels, 7, padding=3, groups=groups)

        if not deploy:
            # Main
            self.conv_7x7 = nn.Conv2d(in_channels, out_channels, 7, padding=3, groups=groups)
            # Directional
            self.conv_7x1 = nn.Conv2d(in_channels, out_channels, (7,1), padding=(3,0), groups=groups)
            self.conv_1x7 = nn.Conv2d(in_channels, out_channels, (1,7), padding=(0,3), groups=groups)
            # Mixed large
            self.conv_7x5 = nn.Conv2d(in_channels, out_channels, (7,5), padding=(3,2), groups=groups)
            self.conv_5x7 = nn.Conv2d(in_channels, out_channels, (5,7), padding=(2,3), groups=groups)
            # Mid
            self.conv_5x5 = nn.Conv2d(in_channels, out_channels, 5, padding=2, groups=groups)
            # Small directional
            self.conv_1x5 = nn.Conv2d(in_channels, out_channels, (1,5), padding=(0,2), groups=groups)
            self.conv_5x1 = nn.Conv2d(in_channels, out_channels, (5,1), padding=(2,0), groups=groups)
            # Sequential branch
            self.conv_1x1_branch = nn.Conv2d(in_channels, in_channels, 1, groups=groups, bias=False)
            self.conv_7x7_branch = nn.Conv2d(in_channels, out_channels, 7, padding=3, groups=groups, bias=False)
        else:
            self._delete_branches()

    def _delete_branches(self):
        for name in [
            'conv_7x7',
            'conv_7x1','conv_1x7',
            'conv_7x5','conv_5x7',
            'conv_5x5',
            'conv_1x5','conv_5x1',
            'conv_1x1_branch','conv_7x7_branch'
        ]:
            if hasattr(self, name):
                delattr(self, name)

    def _pad_to_7x7(self, w):
        _, _, h, w_k = w.shape
        pad_h = 7 - h
        pad_w = 7 - w_k
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return F.pad(w, [pad_left, pad_right, pad_top, pad_bottom])

    def fuse(self, delete_branches=True):
        if self.deploy:
            return
        def get_wb(conv):
            return conv.weight, conv.bias if conv.bias is not None else torch.zeros(self.out_channels, device=conv.weight.device)
        W = torch.zeros_like(self.reparam.weight)
        B = torch.zeros_like(self.reparam.bias)
        def add_branch(w, b):
            nonlocal W, B
            W += w
            B += b
        # Main
        w, b = get_wb(self.conv_7x7)
        add_branch(w, b)
        # Directional
        for conv in [self.conv_7x1, self.conv_1x7]:
            w, b = get_wb(conv)
            add_branch(self._pad_to_7x7(w), b)
        # Mixed large
        for conv in [self.conv_7x5, self.conv_5x7]:
            w, b = get_wb(conv)
            add_branch(self._pad_to_7x7(w), b)
        # Mid
        w, b = get_wb(self.conv_5x5)
        add_branch(self._pad_to_7x7(w), b)
        # Small directional
        for conv in [self.conv_1x5, self.conv_5x1]:
            w, b = get_wb(conv)
            add_branch(self._pad_to_7x7(w), b)
        # Sequential 1x1 → 7x7
        w1 = self.conv_1x1_branch.weight
        w2 = self.conv_7x7_branch.weight
        if self.groups == 1:
            w_seq = F.conv2d(w2, w1.permute(1,0,2,3))
        else:
            w_slices = []
            w1_T = w1.permute(1,0,2,3)
            icpg = self.in_channels // self.groups
            ocpg = self.out_channels // self.groups
            for g in range(self.groups):
                w1_slice = w1_T[:, g*icpg:(g+1)*icpg]
                w2_slice = w2[g*ocpg:(g+1)*ocpg]
                w_slices.append(F.conv2d(w2_slice, w1_slice))
            w_seq = torch.cat(w_slices, dim=0)
        add_branch(w_seq, torch.zeros(self.out_channels, device=w_seq.device))
        self.reparam.weight.data.copy_(W)
        self.reparam.bias.data.copy_(B)
        if delete_branches:
            self._delete_branches()
        self.deploy = True

    def forward(self, x):
        if self.deploy:
            return self.reparam(x)
        return (
            self.conv_7x7(x)
            + self.conv_7x1(x)
            + self.conv_1x7(x)
            + self.conv_7x5(x)
            + self.conv_5x7(x)
            + self.conv_5x5(x)
            + self.conv_1x5(x)
            + self.conv_5x1(x)
            + self.conv_7x7_branch(self.conv_1x1_branch(x))
        )

# # Test RepConv7
# conv = RepConv7(16, 32, groups=16, deploy=False)
# x = torch.randn(1, 16, 64, 64)
# out1 = conv(x)
# print(f"Before fusion: {count_params(conv)} parameters")
# conv.fuse()
# out2 = conv(x)
# print(f"After fusion: {count_params(conv)} parameters")
# print(torch.allclose(out1, out2, atol=1e-5))

@dataclass
class RepAttnConfig:
    dim: int
    num_heads: int = 8
    deploy: bool = False

class RepAttn(nn.Module):
    """ 
    Re-parameterizable Channel Attention Block.
    Sử dụng PyTorch Native SDPA cho tốc độ tối đa và hỗ trợ SVD Post-Training.
    """
    def __init__(self, dim, num_heads=8, deploy=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.deploy = deploy
        
        # 1x1 Convolutions cho QKV và Projection
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Linear Projection (Quá trình này tốn nhiều FLOPs nhất)
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # 2. Reshape cho Channel Attention
        # Chiều Sequence (L) giờ là số kênh mỗi head (C // num_heads)
        # Chiều Embedding (E) giờ là không gian điểm ảnh (H * W)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        # 3. Tính Attention qua backend C++ của PyTorch (Siêu nhanh, tự động scale)
        with torch.amp.autocast('cuda', enabled=False):
            q, k, v = q.float(), k.float(), v.float()
            # PyTorch SDPA tự động tính q @ k.T và chia cho sqrt(h*w)
            attn_out = F.scaled_dot_product_attention(q, k, v)
            
        # 4. Reshape lại về ảnh và out projection
        attn_out = rearrange(attn_out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        out = self.proj(attn_out)
        
        return out.to(x.dtype)

    @torch.no_grad()
    def fuse(self, keep_ratio=0.6):
        """
        Nén mô hình sau khi train (Post-Training Quantization bằng SVD).
        keep_ratio: Tỷ lệ số kênh K được giữ lại (ví dụ 0.5 nghĩa là nén 50% tính toán).
        """
        if self.deploy:
            print("Model already deployed/fused.")
            return

        def apply_svd_to_conv(conv_layer, ratio):
            """Hàm tiện ích để phân rã 1 layer Conv2d thành 2 layer tuần tự."""
            w = conv_layer.weight.data.squeeze() # shape: (out_C, in_C)
            out_c, in_c = w.shape
            
            # Phân rã SVD
            U, S, Vh = torch.linalg.svd(w, full_matrices=False)
            
            # Tính số kênh trung gian K
            K = max(1, int(len(S) * ratio))
            
            # Cắt tỉa (Truncation)
            U_k = U[:, :K]            # (out_C, K)
            S_k = S[:K]               # (K,)
            Vh_k = Vh[:K, :]          # (K, in_C)
            
            # Lớp 1: in_c -> K (Trọng số là S * Vh)
            conv1 = nn.Conv2d(in_c, K, kernel_size=1, bias=False)
            w1 = torch.diag(S_k) @ Vh_k
            conv1.weight.data = w1.view(K, in_c, 1, 1)
            
            # Lớp 2: K -> out_c (Trọng số là U)
            conv2 = nn.Conv2d(K, out_c, kernel_size=1, bias=(conv_layer.bias is not None))
            conv2.weight.data = U_k.view(out_c, K, 1, 1)
            if conv_layer.bias is not None:
                conv2.bias.data = conv_layer.bias.data
                
            return nn.Sequential(conv1, conv2)

        # Thay thế các layer gốc bằng các layer đã nén
        self.qkv = apply_svd_to_conv(self.qkv, ratio=keep_ratio)
        self.proj = apply_svd_to_conv(self.proj, ratio=keep_ratio)
        self.deploy = True
        print(f"[Inference] SVD Fusion completed (keep_ratio={keep_ratio}). qkv & proj splitted.")


# # Test RepAttn
# attn = RepAttn(dim=256, num_heads=8, block_size=14, num_steps=1, pad_type="pre", impl="torch", deploy=False).cuda()
# x = torch.randn(1, 256, 45, 31).cuda()
# out = attn(x)
# print(out.shape)

@dataclass
class FFNConfig:
    dim: int
    expansion_factor: int = 1
    deploy: bool = False

class RepFFN(nn.Module):
    def __init__(self, dim, expansion_factor=1, deploy=False):
        super().__init__()
        hidden_features = int(dim * expansion_factor)
        self.project_in = RepConv3(dim, hidden_features, groups=1, deploy=deploy)
        self.dwconv = RepConv3(hidden_features, hidden_features*2, groups=hidden_features, deploy=deploy)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1)

    @torch.no_grad()
    def fuse(self):
        self.project_in.fuse()
        self.dwconv.fuse()  


    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class SkipConnection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim*2, dim, kernel_size=1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x
    
class RepTransformerBlock(nn.Module):
    def __init__(self, rep_attn_cfg: RepAttnConfig, ffn_cfg: FFNConfig, norm_type='WithBias'):
        super().__init__()
        self.rep_attn = RepAttn(**asdict(rep_attn_cfg))
        self.rep_ffn = RepFFN(**asdict(ffn_cfg))
        self.norm1 = LayerNorm(rep_attn_cfg.dim, norm_type)
        self.norm2 = LayerNorm(rep_attn_cfg.dim, norm_type)

    @torch.no_grad()
    def fuse(self):
        self.rep_attn.fuse()
        self.rep_ffn.fuse()

    def forward(self, x):
        x = x + self.rep_attn(self.norm1(x))
        x = x + self.rep_ffn(self.norm2(x))
        return x
    
class Block(nn.Module):
    def __init__(self, num_block, rep_attn_cfg: RepAttnConfig, ffn_cfg: FFNConfig, norm_type='WithBias'):
        super().__init__()
        self.num_block = num_block
        self.blocks = nn.ModuleList([
            RepTransformerBlock(rep_attn_cfg, ffn_cfg, norm_type) for _ in range(num_block)
        ])

    @torch.no_grad()
    def fuse(self):
        for block in self.blocks:
            block.fuse()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class StyleMapping(nn.Module):
    """Map explicit color gamut and stochastic z into a compact style latent."""

    def __init__(self, color_dim=11, z_dim=128, style_dim=256, hidden_dim=256, num_layers=3):
        super().__init__()
        layers = []
        in_dim = color_dim + z_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.LeakyReLU(0.2, inplace=True)])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, style_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, color_vec, z):
        z = F.normalize(z, dim=1)
        return self.net(torch.cat([color_vec, z], dim=1))


class ModulatedConv2d(nn.Module):
    """StyleGAN-like modulated convolution with optional demodulation."""

    def __init__(self, in_channels, out_channels, kernel_size, style_dim, padding=None, demodulate=True, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2 if padding is None else padding
        self.demodulate = demodulate
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * (1.0 / math.sqrt(in_channels * kernel_size * kernel_size)))
        self.affine = nn.Linear(style_dim, in_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.ones_(self.affine.bias)
        nn.init.normal_(self.affine.weight, mean=0.0, std=0.02)

    def forward(self, x, style):
        b, c, h, w = x.shape
        modulation = self.affine(style).view(b, 1, c, 1, 1)
        weight = self.weight.unsqueeze(0) * modulation
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum(dim=(2, 3, 4)) + 1e-8)
            weight = weight * demod.view(b, self.out_channels, 1, 1, 1)
        weight = weight.reshape(b * self.out_channels, c, self.kernel_size, self.kernel_size)
        x = x.reshape(1, b * c, h, w)
        out = F.conv2d(x, weight, padding=self.padding, groups=b)
        out = out.reshape(b, self.out_channels, out.shape[-2], out.shape[-1])
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


class StyledRefinement(nn.Module):
    """Residual style injection; strength controls how strongly style can modify structure features."""

    def __init__(self, channels, style_dim, strength=1.0):
        super().__init__()
        self.norm = LayerNorm(channels, 'WithBias')
        self.modconv = ModulatedConv2d(channels, channels, kernel_size=3, style_dim=style_dim)
        self.act = nn.GELU()
        self.strength = float(strength)

    def forward(self, x, style):
        residual = self.modconv(self.norm(x), style)
        return x + self.strength * self.act(residual)


class StyleAwareHead(nn.Module):
    def __init__(self, in_channels, style_dim, out_channels=3, deploy=False, last_act=None):
        super().__init__()
        hidden = max(in_channels // 2, 8)
        self.pre = RepConv3(in_channels, hidden, 1, deploy=deploy)
        self.style = StyledRefinement(hidden, style_dim, strength=1.0)
        self.to_rgb = nn.Conv2d(hidden, out_channels, kernel_size=1)
        self.last_act = last_act if last_act is not None else nn.Identity()

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.pre, RepConv3):
            self.pre.fuse()

    def forward(self, x, style):
        x = F.gelu(self.pre(x))
        x = self.style(x, style)
        return self.last_act(self.to_rgb(x))


class MS2I(nn.Module):
    """Structure-style disentangled MS2I generator."""

    def __init__(
        self,
        input_shape=(3, 256, 256),
        deploy=False,
        dims=[48, 96, 192, 384],
        num_blocks=[4, 6, 6, 8],
        num_heads=[1, 2, 2, 4],
        bias=True,
        last_act=None,
        color_dim=11,
        z_dim=128,
        style_dim=256,
        style_strengths=None,
    ):
        super().__init__()
        assert len(dims) == len(num_blocks) == len(num_heads), "Length of dims, num_blocks and num_heads must be the same"
        self.input_shape = input_shape
        self.deploy = deploy
        self.dims = dims
        self.num_blocks = num_blocks
        self.bias = bias
        self.num_heads = num_heads
        self.color_dim = color_dim
        self.z_dim = z_dim
        self.style_dim = style_dim

        self.style_mapping = StyleMapping(color_dim=color_dim, z_dim=z_dim, style_dim=style_dim)

        # Structure encoder: sketch is the only input here, so early features stay layout-dominant.
        self.stem = nn.Conv2d(3, dims[0], kernel_size=7, stride=4, padding=3, bias=bias)

        layers = []
        down_convs = []
        for idx in range(len(dims)):
            attn_cfg, ffn_cfg = self.build_cfg(dims[idx], num_heads[idx])
            block = Block(num_blocks[idx], attn_cfg, ffn_cfg, norm_type='WithBias')
            if idx < len(dims) - 1:
                down_convs.append(DownSample(dims[idx]))
            layers.append(block)
        self.bottleneck = layers[-1]
        self.encoder = nn.ModuleList(layers[:-1])
        self.downsample = nn.ModuleList(down_convs)

        if style_strengths is None:
            style_strengths = [0.15, 0.35, 0.65]
        if len(style_strengths) != len(dims) - 1:
            raise ValueError(f"style_strengths must have {len(dims) - 1} values")

        layers = []
        up_convs = []
        skip_connections = []
        style_layers = []
        for stage, idx in enumerate(range(len(dims) - 2, -1, -1)):
            attn_cfg, ffn_cfg = self.build_cfg(dims[idx], num_heads[idx])
            up_convs.append(UpSample(dims[idx + 1]))
            skip_connections.append(SkipConnection(dims[idx]))
            layers.append(Block(num_blocks[idx], attn_cfg, ffn_cfg, norm_type='WithBias'))
            style_layers.append(StyledRefinement(dims[idx], style_dim, strength=style_strengths[stage]))
        self.decoder = nn.ModuleList(layers)
        self.up_sample = nn.ModuleList(up_convs)
        self.skip = nn.ModuleList(skip_connections)
        self.style_layers = nn.ModuleList(style_layers)

        self.head = StyleAwareHead(dims[0], style_dim, out_channels=3, deploy=deploy, last_act=last_act)

    @torch.no_grad()
    def fuse(self):
        for block in self.encoder:
            block.fuse()
        self.bottleneck.fuse()
        for block in self.decoder:
            block.fuse()
        self.head.fuse()

    def build_cfg(self, dim, head):
        attn_cfg = RepAttnConfig(
            dim=dim,
            num_heads=head,
            deploy=self.deploy,
        )
        ffn_cfg = FFNConfig(dim=dim, expansion_factor=1, deploy=self.deploy)
        return attn_cfg, ffn_cfg

    def _default_color_and_noise(self, sketch):
        b = sketch.size(0)
        color_vec = sketch.new_zeros(b, self.color_dim)
        color_vec[:, COLOR_TO_INDEX['white'] if 'COLOR_TO_INDEX' in globals() else 1] = 1.0
        z = sketch.new_zeros(b, self.z_dim)
        return color_vec, z

    def forward(self, sketch, color_vec=None, z=None, return_latents=False):
        if color_vec is None or z is None:
            default_color, default_z = self._default_color_and_noise(sketch)
            color_vec = default_color if color_vec is None else color_vec
            z = default_z if z is None else z
        color_vec = color_vec.to(device=sketch.device, dtype=sketch.dtype)
        z = z.to(device=sketch.device, dtype=sketch.dtype)
        style = self.style_mapping(color_vec, z)

        x = self.stem(sketch)
        structure_feats = []
        for blk, down in zip(self.encoder, self.downsample):
            x = blk(x)
            structure_feats.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for blk, up, skip, style_layer in zip(self.decoder, self.up_sample, self.skip, self.style_layers):
            x = up(x)
            cur_feat = structure_feats.pop()
            x = skip(x, cur_feat)
            x = blk(x)
            x = style_layer(x, style)

        x = F.interpolate(x, size=self.input_shape[1:], mode='bilinear', align_corners=False)
        out = self.head(x, style)
        if return_latents:
            return out, {'style': style, 'structure': x}
        return out


model_cfg = {
    "input_shape": (3, 256, 256),
    "dims": [32, 64, 128, 256],
    "num_blocks": [1, 2, 2, 4],
    "num_heads": [1, 2, 4, 8],
    "bias": True,
    "last_act": None,
    "deploy": False,
    "color_dim": 11,
    "z_dim": 128,
    "style_dim": 256,
    "style_strengths": [0.15, 0.35, 0.65],
}

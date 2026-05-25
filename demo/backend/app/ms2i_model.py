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

# Global constants...
TARGET_CATEGORIES = [
    "1_top__t_shirt__sweatshirt",
    "0_shirt__blouse",
]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}



@dataclass
class PipelineConfig:
    """Configuration object for paired sketch-real-color samples."""

    real_root: str
    sketch_roots: Dict[str, str]
    color_json_path: Optional[str] = None
    categories: Tuple[str, ...] = tuple(TARGET_CATEGORIES)
    sketch_ratios: Dict[str, float] = None
    seed: int = 42

    def __post_init__(self):
        if self.sketch_ratios is None:
            self.sketch_ratios = {"hed": 0.5, "pencil": 0.3, "canny": 0.2}

        ratio_sum = sum(self.sketch_ratios.values())
        if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6):
            raise ValueError(f"Sketch ratios must sum to 1.0, got {ratio_sum}")


def smart_pad_and_resize(img: Image.Image, target_size: int = 256) -> Image.Image:
    """Pad an image to a square canvas, then resize to the target resolution."""
    w, h = img.size
    max_side = max(w, h)

    # Ensure the square canvas is at least target_size to avoid up/downscale artifacts.
    canvas_side = max(max_side, target_size)

    pad_left = (canvas_side - w) // 2
    pad_top = (canvas_side - h) // 2
    pad_right = canvas_side - w - pad_left
    pad_bottom = canvas_side - h - pad_top

    # Use white padding to match common sketch/clean-background assumptions.
    img_padded = ImageOps.expand(
        img,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=(255, 255, 255),
    )

    if img_padded.size != (target_size, target_size):
        img_padded = img_padded.resize((target_size, target_size), Image.BICUBIC)

    return img_padded


def list_category_images(root: Path, category: str) -> Dict[str, Path]:
    """Return a mapping from filename stem to image path for one category."""
    category_dir = root / category
    if not category_dir.exists():
        return {}

    return {
        p.stem: p
        for p in category_dir.iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    }


COLOR_LABELS = ("Black", "White", "Warm", "Cold")
COLOR_TO_INDEX = {label.lower(): idx for idx, label in enumerate(COLOR_LABELS)}


def color_to_one_hot(label: str) -> List[float]:
    """Map a color-gamut label to a 4D one-hot vector: Black/White/Warm/Cold."""
    key = str(label).strip().lower()
    if key not in COLOR_TO_INDEX:
        raise ValueError(f"Unknown color label '{label}'. Expected one of {COLOR_LABELS}.")
    vec = [0.0] * len(COLOR_LABELS)
    vec[COLOR_TO_INDEX[key]] = 1.0
    return vec


def load_color_lookup(color_json_path: Optional[str]) -> Dict[Tuple[str, str], dict]:
    """Load color metadata keyed by (subfolder/category, original filename stem)."""
    if not color_json_path:
        return {}

    path = Path(color_json_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Color JSON not found: {path}. Provide Data/train2020/(5)_color_classified/color_info.json for training."
        )

    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    lookup = {}
    for item in payload.get('items', []):
        category = item.get('subfolder')
        filename = item.get('filename') or Path(item.get('source_path', '')).name
        label = item.get('label')
        if not category or not filename or not label:
            continue
        stem = Path(filename).stem
        label = str(label).strip().title()
        lookup[(category, stem)] = {
            'color_label': label,
            'color_vec': color_to_one_hot(label),
            'color_hsv': (item.get('h'), item.get('s'), item.get('v')),
        }
    return lookup


def build_gan_pairs(config: PipelineConfig) -> Tuple[List[dict], dict]:
    """Build paired sketch-real samples and attach the 4-gamut color vector."""
    print("\n[1/3] Scanning categories, matching real-sketch files, and attaching color labels...")
    start_time = time.time()

    real_root = Path(config.real_root)
    sketch_roots = {k: Path(v) for k, v in config.sketch_roots.items()}
    color_lookup = load_color_lookup(config.color_json_path)
    rows: List[dict] = []
    skipped_no_sketch = 0
    skipped_no_color = 0

    for category in tqdm(config.categories, desc="Scanning categories"):
        real_images = list_category_images(real_root, category)
        if not real_images:
            continue

        sketch_index = {
            method: list_category_images(root, category)
            for method, root in sketch_roots.items()
        }

        for stem, real_path in real_images.items():
            candidates = {
                method: sketch_index[method][stem]
                for method in sketch_index
                if stem in sketch_index[method]
            }
            if not candidates:
                skipped_no_sketch += 1
                continue

            color_meta = color_lookup.get((category, stem))
            if color_lookup and color_meta is None:
                skipped_no_color += 1
                continue
            if color_meta is None:
                color_meta = {
                    'color_label': 'White',
                    'color_vec': color_to_one_hot('White'),
                    'color_hsv': (None, None, None),
                }

            rows.append(
                {
                    "category": category,
                    "filename_stem": stem,
                    "real_path": str(real_path),
                    "sketch_candidates": candidates,
                    "available_methods": sorted(candidates.keys()),
                    **color_meta,
                }
            )

    print(f"Matched {len(rows)} pairs. Skipped {skipped_no_sketch} without sketches, {skipped_no_color} without color metadata.")
    print("\n[2/3] Assigning sketch methods using ratio targets...")

    rng = random.Random(config.seed)
    total_found = len(rows)
    methods = list(config.sketch_ratios.keys())
    targets = {m: int(total_found * config.sketch_ratios[m]) for m in methods}
    remaining = dict(targets)

    indices = list(range(total_found))
    rng.shuffle(indices)

    for idx in tqdm(indices, desc="Assigning sketch method"):
        row = rows[idx]
        available = row["available_methods"]
        preferred = [m for m in available if remaining.get(m, 0) > 0]
        method = rng.choice(preferred) if preferred else rng.choice(available)

        row["sketch_method"] = method
        row["sketch_path"] = str(row["sketch_candidates"][method])
        if method in remaining:
            remaining[method] -= 1

    final_rows = [
        {
            "category": r["category"],
            "filename_stem": r["filename_stem"],
            "real_path": r["real_path"],
            "sketch_path": r["sketch_path"],
            "sketch_method": r["sketch_method"],
            "color_label": r["color_label"],
            "color_vec": r["color_vec"],
            "color_hsv": r["color_hsv"],
        }
        for r in rows
    ]

    duration = time.time() - start_time
    print(f"Data preparation completed in {duration:.2f}s.")

    summary = {
        "num_pairs": len(final_rows),
        "sketch_method_counts": dict(Counter(r["sketch_method"] for r in final_rows)),
        "color_counts": dict(Counter(r["color_label"] for r in final_rows)),
        "skipped_no_sketch": skipped_no_sketch,
        "skipped_no_color": skipped_no_color,
    }
    return final_rows, summary



class SketchToRealGANDataset(Dataset):
    """PyTorch dataset that loads paired sketch-real images plus a 4D color-gamut vector."""

    def __init__(
        self,
        rows: List[dict],
        image_size: int = 256,
        apply_augmentation: bool = False,
        flip_prob: float = 0.5,
        crop_scale: Tuple[float, float] = (0.9, 1.0),
        max_translate_ratio: float = 0.08,
        max_rotation_deg: float = 10.0,
        scale_range: Tuple[float, float] = (0.95, 1.05),
    ):
        self.rows = rows
        self.image_size = image_size
        self.apply_augmentation = apply_augmentation
        self.flip_prob = flip_prob
        self.crop_scale = crop_scale
        self.max_translate_ratio = max_translate_ratio
        self.max_rotation_deg = max_rotation_deg
        self.scale_range = scale_range

        max_shift_percent = self.max_translate_ratio * 100.0
        self._augment = A.Compose(
            [
                A.HorizontalFlip(p=self.flip_prob),
                A.RandomResizedCrop(
                    size=(self.image_size, self.image_size),
                    scale=self.crop_scale,
                    ratio=(1.0, 1.0),
                    interpolation=cv2.INTER_CUBIC,
                    p=1.0,
                ),
                A.Affine(
                    scale=self.scale_range,
                    translate_percent={"x": (-max_shift_percent / 100.0, max_shift_percent / 100.0), "y": (-max_shift_percent / 100.0, max_shift_percent / 100.0)},
                    rotate=(-self.max_rotation_deg, self.max_rotation_deg),
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=(255, 255, 255),
                    p=1.0,
                ),
            ],
            additional_targets={"real": "image"},
        )

    def _load_img(self, path: str) -> np.ndarray:
        with Image.open(path) as src_img:
            img = src_img.convert("RGB")
        img = smart_pad_and_resize(img, target_size=self.image_size)
        return np.asarray(img, dtype=np.uint8)

    def _apply_spatial_augment_pair(
        self,
        sketch_img: np.ndarray,
        real_img: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.apply_augmentation:
            return sketch_img, real_img

        transformed = self._augment(image=sketch_img, real=real_img)
        return transformed["image"], transformed["real"]

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        arr = img.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        sketch_img = self._load_img(row["sketch_path"])
        real_img = self._load_img(row["real_path"])
        sketch_img, real_img = self._apply_spatial_augment_pair(sketch_img, real_img)
        return {
            "sketch": self._to_tensor(sketch_img),
            "real": self._to_tensor(real_img),
            "color": torch.tensor(row["color_vec"], dtype=torch.float32),
            "color_label": row["color_label"],
            "filename_stem": row["filename_stem"],
        }


def build_gan_dataloader(
    rows: List[dict],
    batch_size: int = 16,
    image_size: int = 256,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    apply_augmentation: bool = False,
) -> DataLoader:
    """Build a DataLoader from paired metadata rows."""
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else max(1, min(4, (os.cpu_count() or 2) // 2))

    dataset = SketchToRealGANDataset(
        rows=rows,
        image_size=image_size,
        apply_augmentation=apply_augmentation,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import random
from pathlib import Path


def preprocess_preview(path: str, target_size: int = 256):
    """Return raw, smart-padded, and resized versions of one image."""
    with Image.open(path) as src_img:
        raw = src_img.convert("RGB")

    w, h = raw.size
    canvas_side = max(w, h, target_size)
    pad_left = (canvas_side - w) // 2
    pad_top = (canvas_side - h) // 2
    pad_right = canvas_side - w - pad_left
    pad_bottom = canvas_side - h - pad_top

    padded = ImageOps.expand(
        raw,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=(255, 255, 255),
    )
    resized = padded.resize((target_size, target_size), Image.BICUBIC)
    return raw, padded, resized


def find_sketch_by_method(row: dict, method: str):
    """Find sketch image path for a specific method using pipeline config roots."""
    pipe_cfg = globals().get("PIPE_CFG")
    if pipe_cfg is None:
        return None

    root = pipe_cfg.sketch_roots.get(method)
    if root is None:
        return None

    category = row["category"]
    stem = row["filename_stem"]
    method_dir = Path(root) / category
    if not method_dir.exists():
        return None

    for ext in IMG_EXTENSIONS:
        candidate = method_dir / f"{stem}{ext}"
        if candidate.exists():
            return str(candidate)
    return None


def tensor_to_uint8_img(t: torch.Tensor) -> np.ndarray:
    """Convert CHW tensor in [-1, 1] to HWC uint8 image."""
    arr = t.detach().cpu().permute(1, 2, 0).numpy()
    arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return arr


def visualize_full_samples(num_samples: int = 3, image_size: int = 256):
    """Visualize full sample information, including all sketch methods and augment result."""
    if plt is None:
        raise ImportError("matplotlib is required for visualization helpers, but it is not installed.")

    gan_rows = globals().get("gan_rows", [])
    if len(gan_rows) == 0:
        print("'gan_rows' chưa có dữ liệu. Hãy chạy Cell 15 trước.")
        return

    if "SketchToRealGANDataset" not in globals():
        print("Dataset class chưa sẵn sàng. Hãy chạy Cell 13 trước.")
        return

    sample_rows = random.sample(gan_rows, k=min(num_samples, len(gan_rows)))

    # Methods come from current config to guarantee complete per-method view.
    pipe_cfg = globals().get("PIPE_CFG")
    method_list = list(pipe_cfg.sketch_roots.keys()) if pipe_cfg is not None else []

    # Build one augmented dataset instance to preview synchronized spatial augmentation.
    preview_dataset = SketchToRealGANDataset(
        rows=sample_rows,
        image_size=image_size,
        apply_augmentation=True,
    )

    # Columns: Real(raw/final), Selected Sketch(raw/final), each method final, augmented pair.
    n_cols = 4 + len(method_list) + 2
    fig, axes = plt.subplots(len(sample_rows), n_cols, figsize=(3.2 * n_cols, 3.0 * len(sample_rows)))
    if len(sample_rows) == 1:
        axes = [axes]

    print(f"Hiển thị {len(sample_rows)} mẫu | số cột: {n_cols}")

    for r, row in enumerate(sample_rows):
        # Real + selected sketch processing previews.
        real_raw, _, real_final = preprocess_preview(row["real_path"], target_size=image_size)
        sketch_raw, _, sketch_final = preprocess_preview(row["sketch_path"], target_size=image_size)

        # Augmented aligned pair preview.
        aug_item = preview_dataset[r]
        aug_sketch = tensor_to_uint8_img(aug_item["sketch"])
        aug_real = tensor_to_uint8_img(aug_item["real"])

        panels = [
            (real_raw, f"Real Raw\\n{Path(row['real_path']).name}"),
            (real_final, "Real Final"),
            (sketch_raw, f"Sketch Raw ({row['sketch_method']})"),
            (sketch_final, "Sketch Final (Selected)"),
        ]

        # Add each sketch type (hed/pencil/canny/...) for full comparison.
        for method in method_list:
            m_path = find_sketch_by_method(row, method)
            if m_path is None:
                blank = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
                panels.append((blank, f"{method.upper()} (missing)"))
            else:
                _, _, m_final = preprocess_preview(m_path, target_size=image_size)
                panels.append((m_final, f"{method.upper()}"))

        panels.append((aug_sketch, "Aug Sketch"))
        panels.append((aug_real, "Aug Real"))

        for c, (img, title) in enumerate(panels):
            axes[r][c].imshow(img)
            axes[r][c].set_title(title, fontsize=9)
            axes[r][c].axis("off")

        # Write sample-level text at the first panel for quick traceability.
        axes[r][0].set_ylabel(
            f"Sample {r + 1}\\ncat: {row['category']}",
            rotation=90,
            fontsize=9,
            labelpad=10,
        )

    plt.suptitle(
        "Full Visualization: Real/Selected Sketch/All Sketch Methods/Augmented Pair",
        y=1.02,
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# Run visualization
#, image_size=256)

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
    block_size: int = 16
    num_steps: int = 2
    pad_type: str = "pre"
    impl: str = "torch"
    deploy: bool = False

class RepAttn(nn.Module):
    """ Re-parameterizable Attention Block using MonarchAttention as the core attention mechanism."""
    def __init__(self, dim, num_heads=8, block_size=14, num_steps=1, pad_type="pre", impl="torch", deploy=False):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.monarch_attn = MonarchAttention(
            block_size=block_size,
            num_steps=num_steps,
            pad_type=pad_type,
            impl=impl
        )
        if deploy:
            self.attn_fn = self.monarch_attn
        else:
            self.attn_fn = self.common_attn
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.deploy = deploy

    def common_attn(self, q, k, v):
        """ Scaled Dot-Product Attention """
        scale = (q.shape[-1]) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return out

    @torch.no_grad()
    def fuse(self):
        if not self.deploy:
            self.attn_fn = self.monarch_attn
            self.deploy = True

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        attn_out = self.attn_fn(q, k, v)
        attn_out = rearrange(attn_out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        out = self.proj(attn_out)
        return out

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

    def __init__(self, color_dim=4, z_dim=128, style_dim=256, hidden_dim=256, num_layers=3):
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
        color_dim=4,
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
            block_size=16,
            num_steps=2,
            pad_type="pre",
            impl="torch",
            deploy=self.deploy,
        )
        ffn_cfg = FFNConfig(dim=dim, expansion_factor=1)
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
    "color_dim": 4,
    "z_dim": 128,
    "style_dim": 256,
    "style_strengths": [0.15, 0.35, 0.65],
}

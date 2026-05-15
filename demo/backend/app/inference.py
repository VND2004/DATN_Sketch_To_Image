from pathlib import Path
from io import BytesIO
import base64

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps

from .ms2i_model import MS2I, model_cfg


COLOR_LABELS = ("Black", "White", "Warm", "Cold")
COLOR_TO_INDEX = {label.lower(): idx for idx, label in enumerate(COLOR_LABELS)}


def color_to_one_hot(label: str) -> list[float]:
    key = str(label).strip().lower()
    if key not in COLOR_TO_INDEX:
        raise ValueError(f"color_label must be one of {COLOR_LABELS}")
    vec = [0.0] * len(COLOR_LABELS)
    vec[COLOR_TO_INDEX[key]] = 1.0
    return vec


def smart_pad_and_resize(img: Image.Image, target_size: int = 256) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    canvas_side = max(w, h, target_size)
    pad_left = (canvas_side - w) // 2
    pad_top = (canvas_side - h) // 2
    pad_right = canvas_side - w - pad_left
    pad_bottom = canvas_side - h - pad_top
    padded = ImageOps.expand(
        img,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=(255, 255, 255),
    )
    return padded.resize((target_size, target_size), Image.BICUBIC)


def image_to_tensor(img: Image.Image, image_size: int = 256) -> torch.Tensor:
    img = smart_pad_and_resize(img, image_size)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def tensor_to_png_bytes(tensor: torch.Tensor) -> bytes:
    image = (tensor.detach().cpu().clamp(-1, 1) + 1.0) / 2.0
    image = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


class MS2IService:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator_cfg = dict(model_cfg)
        generator_cfg["last_act"] = nn.Tanh()
        self.model = MS2I(**generator_cfg).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get("generator_state_dict", ckpt)
        self.model.load_state_dict(state, strict=False)  # strict=False in case some keys differ slightly
        self.model.eval()

    @torch.no_grad()
    def generate(self, sketch: Image.Image, color_label: str, seed: int | None = 7) -> bytes:
        sketch_tensor = image_to_tensor(sketch).unsqueeze(0).to(self.device)
        color_tensor = torch.tensor(
            [color_to_one_hot(color_label)],
            dtype=torch.float32,
            device=self.device,
        )

        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(int(seed))
        z = torch.randn(1, model_cfg["z_dim"], generator=gen, device=self.device)

        fake = self.model(sketch_tensor, color_tensor, z)
        return tensor_to_png_bytes(fake[0])


def png_bytes_to_data_url(payload: bytes) -> str:
    encoded = base64.b64encode(payload).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

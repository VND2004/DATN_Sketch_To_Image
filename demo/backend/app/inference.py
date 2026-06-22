from io import BytesIO
import base64
import os
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

from .ms2i_model import MS2I, model_cfg
from .ms2i_model_base import MS2I as MS2I_Base
from .sketch_fixer_model import load_light_unet_checkpoint


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


def smart_pad_and_resize_gray(img: Image.Image, target_size: int = 256) -> Image.Image:
    w, h = img.size
    canvas_side = max(w, h, target_size)
    pad_left = (canvas_side - w) // 2
    pad_top = (canvas_side - h) // 2
    pad_right = canvas_side - w - pad_left
    pad_bottom = canvas_side - h - pad_top
    padded = ImageOps.expand(
        img,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=255,
    )
    return padded.resize((target_size, target_size), Image.BICUBIC)


def image_to_tensor(img: Image.Image, image_size: int = 256) -> torch.Tensor:
    img = smart_pad_and_resize(img, image_size)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def grayscale_to_tensor(img: Image.Image, image_size: int = 256) -> torch.Tensor:
    img = smart_pad_and_resize_gray(img.convert("L"), image_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_grayscale_image(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().cpu().clamp(0.0, 1.0)
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3:
        image = image.squeeze(0)
    image = (image.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(image, mode="L")


def tensor_to_png_bytes(tensor: torch.Tensor) -> bytes:
    image = (tensor.detach().cpu().clamp(-1, 1) + 1.0) / 2.0
    image = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    image = (tensor.detach().cpu().clamp(-1, 1) + 1.0) / 2.0
    image = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(image)


def pil_to_bgr_array(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_array_to_pil(image: np.ndarray) -> Image.Image:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)


def png_bytes_from_pil(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class MS2IService:
    def __init__(
        self,
        checkpoint_path: str,
        fixer_checkpoint_path: str | None = None,
        fixer_strength: float = 0.7,
        sr_checkpoint_path: str | None = None,
        sr_tile: int = 0,
        model_type: str = "color",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fixer_strength = float(fixer_strength)
        self.model_type = model_type
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get("generator_state_dict", ckpt)
        
        # Determine if the checkpoint is already fused (no branch weights like conv_3x3)
        is_fused = not any("conv_3x3.weight" in k for k in state.keys())
        logger.info(
            "Loaded MS2I checkpoint from %s | keys in checkpoint: %d | Detected as %s (device: %s)",
            checkpoint_path,
            len(state),
            "FUSED" if is_fused else "UNFUSED",
            self.device,
        )

        generator_cfg = dict(model_cfg)
        generator_cfg["last_act"] = nn.Tanh()
        if is_fused:
            generator_cfg["deploy"] = True

        if self.model_type == "color":
            self.model = MS2I(**generator_cfg).to(self.device)
        else:
            base_cfg = {
                k: v for k, v in generator_cfg.items()
                if k in ["input_shape", "dims", "num_blocks", "num_heads", "bias", "last_act", "deploy"]
            }
            self.model = MS2I_Base(**base_cfg).to(self.device)

        self.fixer = None
        if fixer_checkpoint_path:
            self.fixer = load_light_unet_checkpoint(fixer_checkpoint_path, self.device)

        # Load weights and log any mismatches explicitly so problems are not silently ignored.
        load_result = self.model.load_state_dict(state, strict=False)
        if load_result.missing_keys:
            logger.warning(
                "MS2I load_state_dict: %d MISSING keys (these layers keep random init weights!): %s",
                len(load_result.missing_keys),
                load_result.missing_keys,
            ) 
        if load_result.unexpected_keys:
            logger.warning(
                "MS2I load_state_dict: %d UNEXPECTED keys (ignored from checkpoint): %s",
                len(load_result.unexpected_keys),
                load_result.unexpected_keys,
            )
        if not load_result.missing_keys and not load_result.unexpected_keys:
            logger.info("MS2I load_state_dict: all keys matched perfectly.")
        self.model.eval()
        
        if not is_fused:
            logger.info("Fusing MS2I model branches...")
            self.model.fuse()
            logger.info("Fusion complete.")
        else:
            logger.info("Model is already fused, skipping fusion step.")

        self.sr_upsampler = None
        if sr_checkpoint_path:
            logger.info(f"Loading RealESRGAN from {sr_checkpoint_path}...")
            self.sr_upsampler = self._load_super_resolution_model(sr_checkpoint_path, sr_tile)

    def _load_super_resolution_model(self, checkpoint_path: str, sr_tile: int) -> RealESRGANer:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Super-resolution checkpoint not found: {checkpoint_path}. Set MS2I_SR_CHECKPOINT_PATH to a valid Real-ESRGAN x4plus .pth file."
            )

        sr_model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        return RealESRGANer(
            scale=4,
            model_path=checkpoint_path,
            model=sr_model,
            tile=int(sr_tile),
            tile_pad=10,
            pre_pad=0,
            half=self.device.type == "cuda",
        )

    @torch.no_grad()
    def refine_sketch(self, sketch: Image.Image, use_sketch_fixer: bool = True) -> Image.Image:
        if self.fixer is None or not use_sketch_fixer:
            gray_sketch = smart_pad_and_resize_gray(sketch.convert("L"), 256)
            return Image.merge("RGB", (gray_sketch, gray_sketch, gray_sketch))

        sketch_tensor = grayscale_to_tensor(sketch).unsqueeze(0).to(self.device)
        logits = self.fixer(sketch_tensor)
        refined = torch.sigmoid(logits)

        if 0.0 < self.fixer_strength < 1.0:
            refined = torch.lerp(sketch_tensor, refined, self.fixer_strength)

        refined_image = tensor_to_grayscale_image(refined[0])
        return Image.merge("RGB", (refined_image, refined_image, refined_image))

    @torch.no_grad()
    def generate(self, sketch: Image.Image, color_label: str, seed: int | None = 7, use_sketch_fixer: bool = True, use_sr: bool = True) -> dict[str, bytes]:
        logger.info("--- Starting inference generation ---")
        logger.info(f"Parameters: model_type='{self.model_type}', color_label='{color_label}', seed={seed}, fixer_strength={self.fixer_strength}, use_sketch_fixer={use_sketch_fixer}, use_sr={use_sr}")
        
        t0 = time.time()
        refined_sketch = self.refine_sketch(sketch, use_sketch_fixer=use_sketch_fixer)
        t1 = time.time()
        logger.info(f"[Timing] Sketch refinement: {t1 - t0:.3f}s")
        
        sketch_tensor = image_to_tensor(refined_sketch).unsqueeze(0).to(self.device)
        
        t2 = time.time()
        if self.model_type == "color":
            color_tensor = torch.tensor(
                [color_to_one_hot(color_label)],
                dtype=torch.float32,
                device=self.device,
            )

            # z is set to all zeros during inference to prevent color distortion and match training behavior.
            # The seed parameter is kept in the signature for compatibility but not used for generating z.
            z = torch.zeros(1, model_cfg["z_dim"], dtype=torch.float32, device=self.device)

            fake = self.model(sketch_tensor, color_tensor, z)
        else:
            fake = self.model(sketch_tensor)
            
        t3 = time.time()
        logger.info(f"[Timing] MS2I generation: {t3 - t2:.3f}s")

        generated_image = tensor_to_pil_image(fake[0])
        sr_image = generated_image
        if self.sr_upsampler is not None and use_sr:
            t4 = time.time()
            sr_bgr, _ = self.sr_upsampler.enhance(pil_to_bgr_array(generated_image), outscale=4)
            sr_image = bgr_array_to_pil(sr_bgr)
            t5 = time.time()
            logger.info(f"[Timing] RealESRGAN upsampling: {t5 - t4:.3f}s")
            
        logger.info(f"--- Inference complete (Total: {time.time() - t0:.3f}s) ---")
        
        return {
            "refined_sketch": tensor_to_png_bytes(image_to_tensor(refined_sketch)),
            "generated_image_raw": tensor_to_png_bytes(fake[0]),
            "generated_image": png_bytes_from_pil(sr_image),
        }


def png_bytes_to_data_url(payload: bytes) -> str:
    encoded = base64.b64encode(payload).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

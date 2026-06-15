import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .inference import MS2IService, png_bytes_to_data_url


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_PATH = ROOT / "Model" / "Style_v3_11_color" / "60_75.pt" 
DEFAULT_FIXER_CHECKPOINT_PATH = ROOT / "Model" / "sketch_fixer" / "light_unet_sketch_fixer_v1.pth"
DEFAULT_SR_CHECKPOINT_PATH = ROOT / "Model" / "RealESRGAN" / "Real-ESRGAN-x4plus.pth"
CHECKPOINT_PATH = Path(os.getenv("MS2I_CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT_PATH)))
FIXER_CHECKPOINT_PATH = Path(os.getenv("MS2I_FIXER_CHECKPOINT_PATH", str(DEFAULT_FIXER_CHECKPOINT_PATH)))
SR_CHECKPOINT_PATH = Path(os.getenv("MS2I_SR_CHECKPOINT_PATH", str(DEFAULT_SR_CHECKPOINT_PATH)))
FIXER_STRENGTH = float(os.getenv("MS2I_FIXER_STRENGTH", "1"))
SR_TILE = int(os.getenv("MS2I_SR_TILE", "0"))
print(f"Using checkpoint: {CHECKPOINT_PATH}")
print(f"Using sketch fixer checkpoint: {FIXER_CHECKPOINT_PATH}")
print(f"Using super-resolution checkpoint: {SR_CHECKPOINT_PATH}")

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}. Set MS2I_CHECKPOINT_PATH to a valid .pt file."
    )

if not FIXER_CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Sketch fixer checkpoint not found: {FIXER_CHECKPOINT_PATH}. Set MS2I_FIXER_CHECKPOINT_PATH to a valid .pth file."
    )

if not SR_CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Super-resolution checkpoint not found: {SR_CHECKPOINT_PATH}. Set MS2I_SR_CHECKPOINT_PATH to a valid Real-ESRGAN .pth file."
    )

app = FastAPI(title="MS2I-Net Sketch to Image Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for demo
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = MS2IService(
    str(CHECKPOINT_PATH),
    str(FIXER_CHECKPOINT_PATH),
    FIXER_STRENGTH,
    str(SR_CHECKPOINT_PATH),
    SR_TILE,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "checkpoint": str(CHECKPOINT_PATH),
        "sketch_fixer_checkpoint": str(FIXER_CHECKPOINT_PATH),
        "super_resolution_checkpoint": str(SR_CHECKPOINT_PATH),
        "sketch_fixer_strength": service.fixer_strength,
        "device": str(service.device),
        "super_resolution_enabled": service.sr_upsampler is not None,
    }


@app.post("/generate")
async def generate(
    sketch: UploadFile = File(...),
    color_label: str = Form("White"),
    seed: int = Form(7),
    use_sketch_fixer: bool = Form(True),
):
    payload = await sketch.read()
    image = Image.open(BytesIO(payload)).convert("RGB")
    result = service.generate(
        image,
        color_label=color_label,
        seed=seed,
        use_sketch_fixer=use_sketch_fixer,
    )
    return {
        "refined_sketch": png_bytes_to_data_url(result["refined_sketch"]),
        "image": png_bytes_to_data_url(result["generated_image"]),
        "image_raw": png_bytes_to_data_url(result["generated_image_raw"]),
    }

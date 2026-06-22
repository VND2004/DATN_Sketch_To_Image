import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .inference import MS2IService, png_bytes_to_data_url


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_PATH = ROOT / "Model" / "Style_v3_11_color" / "60_75.pt" 
DEFAULT_BASE_CHECKPOINT_PATH = ROOT / "Model" / "MS2I_v2" / "best_60_70.pt"
DEFAULT_FIXER_CHECKPOINT_PATH = ROOT / "Model" / "sketch_fixer" / "light_unet_sketch_fixer_v1.pth"
DEFAULT_SR_CHECKPOINT_PATH = ROOT / "Model" / "RealESRGAN" / "Real-ESRGAN-x4plus.pth"

CHECKPOINT_PATH = Path(os.getenv("MS2I_CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT_PATH)))
BASE_CHECKPOINT_PATH = Path(os.getenv("MS2I_BASE_CHECKPOINT_PATH", str(DEFAULT_BASE_CHECKPOINT_PATH)))
FIXER_CHECKPOINT_PATH = Path(os.getenv("MS2I_FIXER_CHECKPOINT_PATH", str(DEFAULT_FIXER_CHECKPOINT_PATH)))
SR_CHECKPOINT_PATH = Path(os.getenv("MS2I_SR_CHECKPOINT_PATH", str(DEFAULT_SR_CHECKPOINT_PATH)))
FIXER_STRENGTH = float(os.getenv("MS2I_FIXER_STRENGTH", "1"))
SR_TILE = int(os.getenv("MS2I_SR_TILE", "0"))
print(f"Using color checkpoint: {CHECKPOINT_PATH}")
print(f"Using base checkpoint: {BASE_CHECKPOINT_PATH}")
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

service_color = MS2IService(
    str(CHECKPOINT_PATH),
    str(FIXER_CHECKPOINT_PATH),
    FIXER_STRENGTH,
    str(SR_CHECKPOINT_PATH),
    SR_TILE,
    model_type="color"
)

service_base = None
if BASE_CHECKPOINT_PATH.exists():
    service_base = MS2IService(
        str(BASE_CHECKPOINT_PATH),
        str(FIXER_CHECKPOINT_PATH),
        FIXER_STRENGTH,
        str(SR_CHECKPOINT_PATH),
        SR_TILE,
        model_type="base"
    )
else:
    print(f"Warning: Base checkpoint not found at {BASE_CHECKPOINT_PATH}. Base model will not be available.")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "checkpoint": str(CHECKPOINT_PATH),
        "base_checkpoint": str(BASE_CHECKPOINT_PATH),
        "sketch_fixer_checkpoint": str(FIXER_CHECKPOINT_PATH),
        "super_resolution_checkpoint": str(SR_CHECKPOINT_PATH),
        "sketch_fixer_strength": service_color.fixer_strength,
        "device": str(service_color.device),
        "super_resolution_enabled": service_color.sr_upsampler is not None,
        "base_model_enabled": service_base is not None,
    }

@app.post("/generate")
async def generate(
    sketch: UploadFile = File(...),
    color_label: str = Form("White"),
    seed: int = Form(7),
    use_sketch_fixer: bool = Form(True),
    use_sr: bool = Form(True),
    model_type: str = Form("color"),
):
    payload = await sketch.read()
    image = Image.open(BytesIO(payload)).convert("RGB")
    
    svc = service_base if model_type == "base" and service_base is not None else service_color
    
    result = svc.generate(
        image,
        color_label=color_label,
        seed=seed,
        use_sketch_fixer=use_sketch_fixer,
        use_sr=use_sr,
    )
    return {
        "refined_sketch": png_bytes_to_data_url(result["refined_sketch"]),
        "image": png_bytes_to_data_url(result["generated_image"]),
        "image_raw": png_bytes_to_data_url(result["generated_image_raw"]),
    }

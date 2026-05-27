import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .inference import MS2IService, png_bytes_to_data_url


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_PATH = ROOT / "Model" / "Style_v2_color" / "15_30.pt"
DEFAULT_FIXER_CHECKPOINT_PATH = ROOT / "Model" / "sketch_fixer" / "best_light_unet_sketch_fixer_hed_2k4.pth"
CHECKPOINT_PATH = Path(os.getenv("MS2I_CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT_PATH)))
FIXER_CHECKPOINT_PATH = Path(os.getenv("MS2I_FIXER_CHECKPOINT_PATH", str(DEFAULT_FIXER_CHECKPOINT_PATH)))
FIXER_STRENGTH = float(os.getenv("MS2I_FIXER_STRENGTH", "1"))
print(f"Using checkpoint: {CHECKPOINT_PATH}")
print(f"Using sketch fixer checkpoint: {FIXER_CHECKPOINT_PATH}")

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}. Set MS2I_CHECKPOINT_PATH to a valid .pt file."
    )

if not FIXER_CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Sketch fixer checkpoint not found: {FIXER_CHECKPOINT_PATH}. Set MS2I_FIXER_CHECKPOINT_PATH to a valid .pth file."
    )

app = FastAPI(title="MS2I-Net Sketch to Image Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for demo
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = MS2IService(str(CHECKPOINT_PATH), str(FIXER_CHECKPOINT_PATH), FIXER_STRENGTH)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "checkpoint": str(CHECKPOINT_PATH),
        "sketch_fixer_checkpoint": str(FIXER_CHECKPOINT_PATH),
        "sketch_fixer_strength": service.fixer_strength,
        "device": str(service.device),
    }


@app.post("/generate")
async def generate(
    sketch: UploadFile = File(...),
    color_label: str = Form("Warm"),
    seed: int = Form(7),
):
    payload = await sketch.read()
    image = Image.open(BytesIO(payload)).convert("RGB")
    result = service.generate(image, color_label=color_label, seed=seed)
    return {
        "refined_sketch": png_bytes_to_data_url(result["refined_sketch"]),
        "image": png_bytes_to_data_url(result["generated_image"]),
    }

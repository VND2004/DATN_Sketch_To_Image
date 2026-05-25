import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .inference import MS2IService, png_bytes_to_data_url


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_PATH = ROOT / "Model" / "Style_v2_color" / "15_30.pt"
CHECKPOINT_PATH = Path(os.getenv("MS2I_CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT_PATH)))
print(f"Using checkpoint: {CHECKPOINT_PATH}")

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}. Set MS2I_CHECKPOINT_PATH to a valid .pt file."
    )

app = FastAPI(title="MS2I-Net Sketch to Image Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for demo
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = MS2IService(str(CHECKPOINT_PATH))


@app.get("/health")
def health():
    return {"status": "ok", "checkpoint": str(CHECKPOINT_PATH), "device": str(service.device)}


@app.post("/generate")
async def generate(
    sketch: UploadFile = File(...),
    color_label: str = Form("Warm"),
    seed: int = Form(7),
):
    payload = await sketch.read()
    image = Image.open(BytesIO(payload)).convert("RGB")
    result_png = service.generate(image, color_label=color_label, seed=seed)
    return {"image": png_bytes_to_data_url(result_png)}

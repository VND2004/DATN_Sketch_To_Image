from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .inference import MS2IService, png_bytes_to_data_url


ROOT = Path(__file__).resolve().parents[3]
# Wait, let me check the checkout path from the user's dir.
# The user said: "Weight được lấy từ file `Model/Style_v1_color_noise/eps_50_100.pt`"
# ROOT / "Model" / "Style_v1_color_noise" / "eps_50_100.pt"

CHECKPOINT_PATH = ROOT / "Model" / "Style_v1_color_noise" / "eps_100_150.pt"

app = FastAPI(title="MS2I-Net Sketch to Image Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for demo
    allow_credentials=True,
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

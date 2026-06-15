# Demo UI Sketch to Image cho MS2I-Net

Tài liệu này hướng dẫn tạo demo web để:

- Vẽ sketch trực tiếp trên canvas.
- Chọn gam màu: `Black`, `White`, `Warm`, `Cold`.
- Bấm nút generate để gọi model MS2I-Net.
- Hiển thị ảnh thật sinh ra ở khung bên cạnh.
- Ảnh thật sẽ đi qua Real-ESRGAN x4plus trước khi hiển thị.

Pipeline bám theo model notebook `notebook/MS2I-Net-add-attribute.ipynb`:

- Weight được lấy từ file `Model/Style_v1_color_noise/eps_50_100.pt`
- Real-ESRGAN sẽ dùng weight có sẵn tại `Model/RealESRGAN/Real-ESRGAN-x4plus.pth`

## 1. Kiến trúc demo đề xuất

Dùng React cho UI và FastAPI cho backend inference.

```text
demo/
  backend/
    app/
      main.py              # API nhận sketch + color, trả ảnh generated
      inference.py         # load model, preprocess, postprocess
      ms2i_model.py        # copy class MS2I và các block liên quan từ notebook
    requirements.txt
  frontend/
    package.json
    src/
      App.jsx              # canvas, chọn màu, nút generate, preview kết quả
      main.jsx
      styles.css
```

Lý do tách backend/frontend:

- React chạy tốt phần canvas và tương tác người dùng.
- PyTorch/model nên đặt ở backend Python để load checkpoint một lần và infer nhanh.
- Frontend chỉ gửi ảnh sketch dạng PNG/base64 hoặc multipart file.

## 2. Chuẩn bị model cho backend

Từ notebook cần tách các phần sau sang `demo/backend/app/ms2i_model.py`:

- Các import model: `torch`, `torch.nn as nn`, `torch.nn.functional as F`, `math`.
- Các lớp nền mà `MS2I` phụ thuộc, ví dụ:
  - `LayerNorm`
  - `BiasFree_LayerNorm`
  - `WithBias_LayerNorm`
  - `RepConv`, `RepConv3`, `RepAttn`, các block encoder/decoder nếu có
  - `StyleMapping`
  - `ModulatedConv2d`
  - `StyledRefinement`
  - `StyleAwareHead`
  - `MS2I`
- `model_cfg`:

```python
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
```

Khi load inference, set:

```python
generator_cfg = dict(model_cfg)
generator_cfg["last_act"] = nn.Tanh()
model = MS2I(**generator_cfg)
```

Checkpoint ưu tiên:

```text
../../Model/Style_v1_color_noise/eps_50_100.pt```

Nếu checkpoint này không đúng phiên bản model, thử các checkpoint khác:

```text
../../Model/Style_v1_color_noise/eps_0_50.pt
../../Model/3/last.pt
```

## 3. Backend FastAPI

### `demo/backend/requirements.txt`

```text
fastapi
uvicorn[standard]
python-multipart
pillow
numpy
torch
torchvision
realesrgan
basicsr
opencv-python-headless
```

Nếu chưa có weight Real-ESRGAN, tải file x4plus và đặt vào:

```text
Model/RealESRGAN/Real-ESRGAN-x4plus.pth
```

Hoặc set biến môi trường `MS2I_SR_CHECKPOINT_PATH` trỏ tới file `.pth` bạn đang dùng.

### `demo/backend/app/inference.py`

```python
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
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.no_grad()
    def generate(self, sketch: Image.Image, color_label: str, seed: int | None = 7) -> bytes:
        sketch_tensor = image_to_tensor(sketch).unsqueeze(0).to(self.device)
        color_tensor = torch.tensor(
            [color_to_one_hot(color_label)],
            dtype=torch.float32,
            device=self.device,
        )

        # z is set to all zeros during inference to prevent color distortion
        z = torch.zeros(1, model_cfg["z_dim"], dtype=torch.float32, device=self.device)

        fake = self.model(sketch_tensor, color_tensor, z)
        return tensor_to_png_bytes(fake[0])


def png_bytes_to_data_url(payload: bytes) -> str:
    encoded = base64.b64encode(payload).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
```

### `demo/backend/app/main.py`

```python
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .inference import MS2IService, png_bytes_to_data_url


ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_PATH = ROOT / "Model" / "Best" / "best.pt"

app = FastAPI(title="MS2I-Net Sketch to Image Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = MS2IService(str(CHECKPOINT_PATH))


@app.get("/health")
def health():
    return {"status": "ok", "checkpoint": str(CHECKPOINT_PATH)}


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
```

Chạy backend:

```bash
cd demo/backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Kiểm tra:

```bash
curl http://127.0.0.1:8000/health
```

## 4. Frontend React

Tạo app:

```bash
cd demo
npm create vite@latest frontend -- --template react
cd frontend
npm install
npm run dev
```

Cài thêm icon nếu muốn:

```bash
npm install lucide-react
```

### `demo/frontend/src/App.jsx`

```jsx
import { useEffect, useRef, useState } from "react";
import { Brush, Eraser, RotateCcw, Sparkles } from "lucide-react";
import "./styles.css";

const API_URL = "http://127.0.0.1:8000/generate";
const COLORS = ["Black", "White", "Warm", "Cold"];

export default function App() {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [tool, setTool] = useState("brush");
  const [brushSize, setBrushSize] = useState(8);
  const [colorLabel, setColorLabel] = useState("Warm");
  const [seed, setSeed] = useState(7);
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
  }, []);

  const getPoint = (event) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const source = event.touches?.[0] ?? event;
    return {
      x: ((source.clientX - rect.left) / rect.width) * canvas.width,
      y: ((source.clientY - rect.top) / rect.height) * canvas.height,
    };
  };

  const startDrawing = (event) => {
    event.preventDefault();
    const ctx = canvasRef.current.getContext("2d");
    const point = getPoint(event);
    ctx.beginPath();
    ctx.moveTo(point.x, point.y);
    setIsDrawing(true);
  };

  const draw = (event) => {
    if (!isDrawing) return;
    event.preventDefault();
    const ctx = canvasRef.current.getContext("2d");
    const point = getPoint(event);
    ctx.strokeStyle = tool === "eraser" ? "#ffffff" : "#111111";
    ctx.lineWidth = brushSize;
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setResult("");
  };

  const canvasToBlob = () => {
    return new Promise((resolve) => {
      canvasRef.current.toBlob(resolve, "image/png");
    });
  };

  const generate = async () => {
    setLoading(true);
    try {
      const blob = await canvasToBlob();
      const formData = new FormData();
      formData.append("sketch", blob, "sketch.png");
      formData.append("color_label", colorLabel);
      formData.append("seed", String(seed));

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(`Generate failed: ${response.status}`);
      }
      const data = await response.json();
      setResult(data.image);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="app">
      <section className="workspace">
        <div className="panel">
          <div className="toolbar">
            <button className={tool === "brush" ? "active" : ""} onClick={() => setTool("brush")} title="Brush">
              <Brush size={18} />
            </button>
            <button className={tool === "eraser" ? "active" : ""} onClick={() => setTool("eraser")} title="Eraser">
              <Eraser size={18} />
            </button>
            <button onClick={clearCanvas} title="Clear">
              <RotateCcw size={18} />
            </button>
            <input
              type="range"
              min="2"
              max="28"
              value={brushSize}
              onChange={(event) => setBrushSize(Number(event.target.value))}
            />
          </div>

          <canvas
            ref={canvasRef}
            width="512"
            height="512"
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing}
            onTouchStart={startDrawing}
            onTouchMove={draw}
            onTouchEnd={stopDrawing}
          />
        </div>

        <div className="panel result-panel">
          {result ? <img src={result} alt="Generated result" /> : <div className="empty">Generated image</div>}
        </div>
      </section>

      <section className="controls">
        <div className="segmented">
          {COLORS.map((item) => (
            <button
              key={item}
              className={colorLabel === item ? "active" : ""}
              onClick={() => setColorLabel(item)}
            >
              {item}
            </button>
          ))}
        </div>

        <label className="seed">
          Seed
          <input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
        </label>

        <button className="generate" onClick={generate} disabled={loading}>
          <Sparkles size={18} />
          {loading ? "Generating..." : "Generate"}
        </button>
      </section>
    </main>
  );
}
```

### `demo/frontend/src/styles.css`

```css
* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f3f5f7;
  color: #17202a;
}

button,
input {
  font: inherit;
}

.app {
  min-height: 100vh;
  padding: 24px;
  display: grid;
  grid-template-rows: 1fr auto;
  gap: 18px;
}

.workspace {
  display: grid;
  grid-template-columns: minmax(320px, 1fr) minmax(320px, 1fr);
  gap: 18px;
  min-height: 0;
}

.panel {
  background: #ffffff;
  border: 1px solid #d8dee6;
  border-radius: 8px;
  padding: 14px;
  display: grid;
  grid-template-rows: auto 1fr;
  min-height: 0;
}

.toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
}

.toolbar button,
.segmented button,
.generate {
  border: 1px solid #c7d0da;
  background: #ffffff;
  color: #17202a;
  border-radius: 8px;
  height: 38px;
  padding: 0 12px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  cursor: pointer;
}

.toolbar button {
  width: 38px;
  padding: 0;
}

.toolbar button.active,
.segmented button.active {
  background: #17202a;
  border-color: #17202a;
  color: #ffffff;
}

.toolbar input {
  width: 140px;
}

canvas {
  width: 100%;
  aspect-ratio: 1 / 1;
  background: #ffffff;
  border: 1px solid #d8dee6;
  border-radius: 8px;
  touch-action: none;
}

.result-panel {
  grid-template-rows: 1fr;
}

.result-panel img,
.empty {
  width: 100%;
  aspect-ratio: 1 / 1;
  border: 1px solid #d8dee6;
  border-radius: 8px;
  background: #eef2f6;
}

.result-panel img {
  object-fit: contain;
}

.empty {
  display: grid;
  place-items: center;
  color: #6b7886;
}

.controls {
  background: #ffffff;
  border: 1px solid #d8dee6;
  border-radius: 8px;
  padding: 12px;
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.segmented {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.seed {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: #405060;
}

.seed input {
  width: 92px;
  height: 38px;
  border: 1px solid #c7d0da;
  border-radius: 8px;
  padding: 0 10px;
}

.generate {
  margin-left: auto;
  background: #166534;
  border-color: #166534;
  color: #ffffff;
  min-width: 138px;
}

.generate:disabled {
  opacity: 0.6;
  cursor: wait;
}

@media (max-width: 860px) {
  .app {
    padding: 14px;
  }

  .workspace {
    grid-template-columns: 1fr;
  }

  .generate {
    margin-left: 0;
    width: 100%;
  }
}
```

## 5. Luồng request/response

Frontend gửi:

```text
POST /generate
Content-Type: multipart/form-data

sketch: sketch.png
color_label: Warm
seed: 7
```

Backend trả:

```json
{
  "image": "data:image/png;base64,..."
}
```

React gán trực tiếp `image` vào `src` của thẻ `<img>`.

## 6. Checklist triển khai

1. Tạo `demo/backend/app/ms2i_model.py` từ các cell định nghĩa model trong notebook.
2. Tạo `inference.py`, `main.py`, `requirements.txt` như trên.
3. Chạy backend tại `http://127.0.0.1:8000`.
4. Tạo React app bằng Vite trong `demo/frontend`.
5. Thay `App.jsx` và `styles.css`.
6. Chạy frontend tại `http://localhost:5173`.
7. Vẽ sketch, chọn gam màu, bấm `Generate`.

## 7. Lưu ý quan trọng

- Canvas nên giữ nền trắng, nét vẽ đen, vì notebook xử lý sketch theo giả định nền trắng.
- Backend nên load model một lần khi app start, không load lại mỗi request.
- Nếu muốn output ổn định, giữ seed cố định.
- Nếu muốn mỗi lần generate ra texture khác nhau, đổi seed hoặc random seed ở frontend.
- Nếu GPU khả dụng, FastAPI sẽ tự dùng `cuda`; nếu không, CPU vẫn chạy được nhưng chậm hơn.
- Nếu checkpoint báo lỗi `missing keys` hoặc `unexpected keys`, checkpoint không khớp code model. Khi đó cần kiểm tra lại đúng phiên bản `MS2I` và `model_cfg` tương ứng với checkpoint.

## 8. Cải tiến sau khi demo chạy được

- Thêm nút upload sketch có sẵn.
- Thêm download ảnh generated.
- Thêm preview nhiều seed cùng lúc.
- Thêm trạng thái backend đang dùng `cuda` hay `cpu` ở `/health`.
- Export model sang TorchScript hoặc ONNX nếu cần tối ưu tốc độ.

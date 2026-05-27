import { useEffect, useRef, useState } from "react";
import { Brush, Eraser, ImageUp, RotateCcw, Sparkles } from "lucide-react";
import "./styles.css";

const API_URL = "http://127.0.0.1:8000/generate";
const COLORS = ["Black", "White", "Warm", "Cold"];

export default function App() {
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [tool, setTool] = useState("brush");
  const [brushSize, setBrushSize] = useState(8);
  const [colorLabel, setColorLabel] = useState("Warm");
  const [seed, setSeed] = useState(7);
  const [refinedSketch, setRefinedSketch] = useState("");
  const [result, setResult] = useState("");
  const [previewMode, setPreviewMode] = useState("result");
  const [loading, setLoading] = useState(false);

  const previewImage = previewMode === "refined" ? refinedSketch : result;
  const previewTitle = previewMode === "refined" ? "Sketch đã hoàn thiện nét" : "Ảnh thật";
  const previewAlt = previewMode === "refined" ? "Refined sketch" : "Generated realistic image";

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
    setRefinedSketch("");
    setResult("");
    setPreviewMode("result");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const loadSketchFile = (file) => {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const scale = Math.min(canvas.width / image.width, canvas.height / image.height);
        const width = image.width * scale;
        const height = image.height * scale;
        const offsetX = (canvas.width - width) / 2;
        const offsetY = (canvas.height - height) / 2;

        ctx.drawImage(image, offsetX, offsetY, width, height);
        URL.revokeObjectURL(image.src);
        setRefinedSketch("");
        setResult("");
        setPreviewMode("result");
        resolve();
      };
      image.onerror = () => {
        URL.revokeObjectURL(image.src);
        reject(new Error("Could not load the selected image"));
      };
      image.src = URL.createObjectURL(file);
    });
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    await loadSketchFile(file);
    event.target.value = "";
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
      setRefinedSketch(data.refined_sketch);
      setResult(data.image);
      setPreviewMode("result");
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
          <div className="toolbar placeholder" aria-hidden="true" />
          {previewImage ? (
            <figure className="single-preview">
              <img src={previewImage} alt={previewAlt} />
              <figcaption>{previewTitle}</figcaption>
            </figure>
          ) : (
            <div className="empty">Ảnh thật sẽ hiển thị ở đây</div>
          )}
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

        <div className="upload-row">
          <label className="upload-button">
            <ImageUp size={18} />
            Upload sketch
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
            />
          </label>
          <span className="upload-hint">Chọn ảnh sketch từ máy để nạp vào khung vẽ</span>
        </div>

        <button
          className="preview-toggle"
          onClick={() => setPreviewMode((current) => (current === "result" ? "refined" : "result"))}
          disabled={!refinedSketch && !result}
        >
          {previewMode === "refined" ? "Xem ảnh thật" : "Xem sketch hoàn thiện"}
        </button>

        <button className="generate" onClick={generate} disabled={loading}>
          <Sparkles size={18} />
          {loading ? "Generating..." : "Generate"}
        </button>
      </section>
    </main>
  );
}

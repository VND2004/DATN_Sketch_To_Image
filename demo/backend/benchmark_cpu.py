import os
import time
import argparse
import logging
from pathlib import Path
from PIL import Image
import numpy as np

# Vô hiệu hóa GPU để ép chạy trên CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Import sau khi set biến môi trường CUDA để đảm bảo PyTorch không gọi GPU
import torch
from app.inference import MS2IService

# Lấy thư mục gốc (DATN_Sketch_To_Image)
ROOT_DIR = Path(__file__).resolve().parents[2]

# Khai báo danh sách các model để đánh giá
MODELS = {
    "color": {
        "path": str(ROOT_DIR / "Model" / "Style_v3_11_color" / "60_75.pt"),
        "type": "color"
    },
    "base": {
        "path": str(ROOT_DIR / "Model" / "MS2I_v2" / "best_60_70.pt"),
        "type": "base"
    }
}

def create_dummy_sketch(size=256):
    """Tạo một ảnh sketch giả định để phục vụ benchmark."""
    img_array = np.ones((size, size, 3), dtype=np.uint8) * 255
    # Vẽ vài nét đen giả lập sketch
    img_array[100:150, 100:150] = 0
    return Image.fromarray(img_array)

def run_benchmark(model_name, iterations=100, use_fixer=False, use_sr=False):
    if model_name not in MODELS:
        logger.error(f"Model '{model_name}' không tồn tại trong danh sách.")
        logger.info(f"Các model khả dụng: {', '.join(MODELS.keys())}")
        return

    model_info = MODELS[model_name]
    checkpoint_path = model_info["path"]
    model_type = model_info["type"]

    if not os.path.exists(checkpoint_path):
        logger.error(f"Không tìm thấy file checkpoint tại: {checkpoint_path}")
        logger.error("Vui lòng cập nhật đường dẫn chính xác trong từ điển MODELS của file benchmark này.")
        return

    logger.info(f"Đang tải model '{model_name}' ({model_type}) từ {checkpoint_path}...")
    logger.info("Chế độ: CPU")
    
    # Lấy đường dẫn của fixer và SR nếu được yêu cầu
    fixer_path = str(ROOT_DIR / "Model" / "sketch_fixer" / "light_unet_sketch_fixer_v1.pth") if use_fixer else None
    sr_path = str(ROOT_DIR / "Model" / "RealESRGAN" / "Real-ESRGAN-x4plus.pth") if use_sr else None

    # Tắt logger của app.inference để không in ra quá nhiều text rác trong quá trình lặp 100 lần
    logging.getLogger('app.inference').setLevel(logging.WARNING)

    # Khởi tạo model service
    service = MS2IService(
        checkpoint_path=checkpoint_path,
        fixer_checkpoint_path=fixer_path if use_fixer and os.path.exists(fixer_path) else None,
        sr_checkpoint_path=sr_path if use_sr and os.path.exists(sr_path) else None,
        model_type=model_type
    )

    # Ép service và các model con dùng CPU
    service.device = torch.device("cpu")
    service.model.to(service.device)
    if service.fixer:
        service.fixer.to(service.device)
    if service.sr_upsampler:
        service.sr_upsampler.model.to(service.device)

    dummy_sketch = create_dummy_sketch()
    color_label = "White"

    logger.info("Bắt đầu Warmup (chạy nháp 3 lần để model được khởi tạo hoàn toàn)...")
    for _ in range(3):
        service.generate(dummy_sketch, color_label=color_label, use_sketch_fixer=use_fixer, use_sr=use_sr)

    logger.info(f"\nBắt đầu Benchmark {iterations} lần...")
    inference_times = []

    for i in range(iterations):
        start_t = time.time()
        # Gọi hàm generate để tạo ảnh
        service.generate(dummy_sketch, color_label=color_label, use_sketch_fixer=use_fixer, use_sr=use_sr)
        end_t = time.time()
        
        iter_time = end_t - start_t
        inference_times.append(iter_time)

        if (i + 1) % 10 == 0:
            logger.info(f" -> Đã hoàn thành {i + 1}/{iterations} lặp. Tốc độ lần này: {iter_time:.4f}s")

    # Tính toán kết quả
    avg_time = sum(inference_times) / iterations
    fps = 1.0 / avg_time

    print("\n" + "="*50)
    print("                KẾT QUẢ BENCHMARK")
    print("="*50)
    print(f"Tên mô hình            : {model_name}")
    print(f"Số lần chạy            : {iterations}")
    print(f"Thiết bị               : CPU")
    print(f"Dùng Sketch Fixer      : {'Có' if use_fixer else 'Không'}")
    print(f"Dùng Super Resolution  : {'Có' if use_sr else 'Không'}")
    print(f"Average Inference Time : {avg_time:.4f} giây/ảnh")
    print(f"FPS (Frames Per Second): {fps:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Công cụ Benchmark tốc độ Inference trên CPU")
    parser.add_argument("--model", type=str, required=True, 
                        help=f"Tên mô hình cần test. Các lựa chọn: {', '.join(MODELS.keys())}")
    parser.add_argument("--iters", type=int, default=100, help="Số vòng lặp (mặc định: 100)")
    parser.add_argument("--use_fixer", action="store_true", help="Bật module Sketch Fixer trong quá trình test")
    parser.add_argument("--use_sr", action="store_true", help="Bật module Super Resolution trong quá trình test")

    args = parser.parse_args()
    
    run_benchmark(
        model_name=args.model,
        iterations=args.iters,
        use_fixer=args.use_fixer,
        use_sr=args.use_sr
    )

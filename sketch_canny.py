import os
import shutil
import cv2
from tqdm import tqdm

input_root = r"D:\THO\Bach_Khoa\Thesis\Data\val_test2020\(3)_images_filtered"
output_root = r"D:\THO\Bach_Khoa\Thesis\Data\val_test2020\(4)_sketch_canny"

# ====== RESET OUTPUT FOLDER ======
if os.path.exists(output_root):
    shutil.rmtree(output_root)
os.makedirs(output_root)

# Tham số Canny
low_threshold = 100
high_threshold = 200
kernel_size = 5

# ====== STEP 1: Collect all image paths ======
image_paths = []

for subdir, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_paths.append(os.path.join(subdir, file))

# ====== STEP 2: Process with tqdm ======
for input_path in tqdm(image_paths, desc="Processing images"):
    # Tạo output path tương ứng
    relative_path = os.path.relpath(input_path, input_root)
    output_path = os.path.join(output_root, relative_path)

    output_subdir = os.path.dirname(output_path)
    os.makedirs(output_subdir, exist_ok=True)

    # Đọc ảnh
    img = cv2.imread(input_path)
    if img is None:
        continue

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Canny
    edges = cv2.Canny(blur, low_threshold, high_threshold)

    # Đảo đen trắng: nét đen trên nền trắng
    edges = 255 - edges

    # Lưu
    cv2.imwrite(output_path, edges)
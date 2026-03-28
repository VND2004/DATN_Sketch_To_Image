import os
import shutil
import cv2
from tqdm import tqdm

input_root = r"D:\THO\Bach_Khoa\Thesis\Data\val_test2020\(3)_images_filtered"
output_root = r"D:\THO\Bach_Khoa\Thesis\Data\val_test2020\(4)_sketch_pencil"

# ====== RESET OUTPUT FOLDER ======
if os.path.exists(output_root):
    shutil.rmtree(output_root)
os.makedirs(output_root)

# Tham số Pencil Sketch
kernel_size = 21

# ====== Pencil Sketch Function ======
def pencil_sketch(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert
    inverted = 255 - gray

    # Blur
    blur = cv2.GaussianBlur(inverted, (kernel_size, kernel_size), 0)

    # Invert blur
    inverted_blur = 255 - blur

    # Divide để tạo sketch
    sketch = cv2.divide(gray, inverted_blur, scale=256)

    return sketch

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

    # ====== Pencil Sketch ======
    sketch = pencil_sketch(img)

    # Lưu
    cv2.imwrite(output_path, sketch)
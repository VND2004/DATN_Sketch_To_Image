import os
import cv2
from tqdm import tqdm
import shutil

# ====== CONFIG ======
input_root = r"D:\THO\Bach_Khoa\Thesis\Data\train2020\train_no_bg_blur"   # folder gốc (có folder con)
mask_folder = r"D:\THO\Bach_Khoa\Thesis\Data\train2020\masks_filtered"
output_root = r"D:\THO\Bach_Khoa\Thesis\Data\train2020\images_filtered"

min_size = 100

# ====== RESET OUTPUT FOLDER ======
if os.path.exists(output_root):
    shutil.rmtree(output_root)
os.makedirs(output_root)

# ====== LOAD MASK FILENAMES ======
mask_names = set(os.listdir(mask_folder))  # dùng set để check nhanh

# ====== LOG STATS ======
total = 0
kept = 0
removed = 0
removed_not_in_mask = 0
removed_small = 0
error_count = 0

# ====== WALK THROUGH SUBFOLDERS ======
all_images = []

for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            full_path = os.path.join(root, file)
            all_images.append(full_path)

# ====== PROCESS ======
for img_path in tqdm(all_images, desc="Filtering images"):
    total += 1

    img_name = os.path.basename(img_path)

    # --- check có trong mask_filtered ---
    if img_name not in mask_names:
        removed += 1
        removed_not_in_mask += 1
        continue

    # --- đọc ảnh ---
    img = cv2.imread(img_path)
    if img is None:
        error_count += 1
        continue

    h, w = img.shape[:2]

    # --- check size ---
    if h < min_size or w < min_size:
        removed += 1
        removed_small += 1
        continue

    # --- tạo path output giữ nguyên structure ---
    relative_path = os.path.relpath(img_path, input_root)
    save_path = os.path.join(output_root, relative_path)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cv2.imwrite(save_path, img)
    kept += 1

# ====== PRINT SUMMARY ======
print("\n===== FILTER SUMMARY =====")
print(f"Total images           : {total}")
print(f"Kept images            : {kept}")
print(f"Removed images         : {removed}")
print(f"  - Not in mask        : {removed_not_in_mask}")
print(f"  - Too small          : {removed_small}")
print(f"Error images           : {error_count}")
print(f"Kept ratio             : {kept / total:.2%}" if total > 0 else "N/A")
import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# ====== CONFIG ======
input_folder = r"D:\THO\Bach_Khoa\Thesis\Data\train2020\masks"
filtered_folder = r"D:\THO\Bach_Khoa\Thesis\Data\train2020\masks_filtered"
removed_folder = r"D:\THO\Bach_Khoa\Thesis\Data\train2020\masks_removed"

coverage_threshold = 0.02  # 2%

# ====== RESET OUTPUT FOLDERS ======
for folder in [filtered_folder, removed_folder]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# ====== GET IMAGE LIST ======
image_files = [f for f in os.listdir(input_folder)
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

# ====== LOG STATS ======
total = len(image_files)
kept = 0
removed = 0
removed_low_coverage = 0
error_count = 0

# ====== PROCESS ======
for img_name in tqdm(image_files, desc="Filtering masks"):
    img_path = os.path.join(input_folder, img_name)

    mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        error_count += 1
        continue

    h, w = mask.shape

    # --- Compute coverage ---
    white_pixels = np.sum(mask > 127)
    total_pixels = h * w
    coverage = white_pixels / total_pixels

    # --- Filter ---
    if coverage <= coverage_threshold:
        save_path = os.path.join(removed_folder, img_name)
        removed += 1
        removed_low_coverage += 1
    else:
        save_path = os.path.join(filtered_folder, img_name)
        kept += 1

    cv2.imwrite(save_path, mask)

# ====== PRINT SUMMARY ======
print("\n===== FILTER SUMMARY =====")
print(f"Total images        : {total}")
print(f"Kept images         : {kept}")
print(f"Removed images      : {removed}")
print(f"  - Low coverage    : {removed_low_coverage}")
print(f"Error images        : {error_count}")
print(f"Kept ratio          : {kept / total:.2%}" if total > 0 else "N/A")
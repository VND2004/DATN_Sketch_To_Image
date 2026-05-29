import argparse
import os
import random

import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

# ====== CONFIG ======
# INPUT_ROOT = r"D:\Git\DATN_Sketch_To_Image\Data\train2020\sketch_category_filtered\(2)_sketch_hed"
# OUTPUT_ROOT = r"D:\Git\DATN_Sketch_To_Image\sketch_degraded_hed"

INPUT_ROOT = r"/mnt/d/Git/DATN_Sketch_To_Image/Data/train2020/sketch_category_filtered/(2)_sketch_hed"
OUTPUT_ROOT = r"/mnt/d/Git/DATN_Sketch_To_Image/sketch_degraded_hed"

SELECTED_FOLDERS = [
    "10_dress",
    "8_skirt",
    "43_ruffle",
    "1_top__t_shirt__sweatshirt",
    "0_shirt__blouse",
    "4_jacket",
    "9_coat",
    "2_sweater",
    "3_cardigan",
    "5_vest",
    "6_pants",
    "7_shorts",
]

ERASE_PROB = 0.0003
MIN_PATCH = 10
MAX_PATCH = 30
EDGE_THRESHOLD = 230
FADE_MODE = False
MAX_IMAGES_PER_FOLDER = 200
# ====================


def _build_output_name(relative_path):
    return f"hed__{os.path.basename(relative_path)}"


def _get_unique_output_path(output_root, output_name):
    save_path = os.path.join(output_root, output_name)
    if not os.path.exists(save_path):
        return save_path

    base_name, extension = os.path.splitext(output_name)
    suffix = 1
    while True:
        candidate_path = os.path.join(output_root, f"{base_name}_{suffix}{extension}")
        if not os.path.exists(candidate_path):
            return candidate_path
        suffix += 1


def degrade_sketch_grayscale_square(
    image,
    erase_prob=0.02,
    min_patch=3,
    max_patch=10,
    edge_threshold=220,
    fade_mode=True,
):
    """Degrade sketch grayscale bằng cách làm mất nét với patch hình vuông."""

    degraded = image.copy().astype(np.float32)
    h, w = degraded.shape

    edge_pixels = np.argwhere(degraded < edge_threshold)
    num_pixels = len(edge_pixels)

    if num_pixels == 0:
        return image

    num_erase = int(num_pixels * erase_prob)
    if num_erase <= 0:
        return image

    selected_indices = np.random.choice(num_pixels, num_erase, replace=False)

    for idx in selected_indices:
        y, x = edge_pixels[idx]

        patch_size = random.randint(min_patch, max_patch)
        half = patch_size // 2

        x1 = max(0, x - half)
        y1 = max(0, y - half)
        x2 = min(w, x + half)
        y2 = min(h, y + half)

        if fade_mode:
            alpha = random.uniform(0.4, 0.9)
            degraded[y1:y2, x1:x2] = degraded[y1:y2, x1:x2] * (1 - alpha) + 255 * alpha
        else:
            degraded[y1:y2, x1:x2] = 255

    return np.clip(degraded, 0, 255).astype(np.uint8)


def process_folder(
    input_root,
    output_root,
    selected_folders=None,
    erase_prob=ERASE_PROB,
    min_patch=MIN_PATCH,
    max_patch=MAX_PATCH,
    edge_threshold=EDGE_THRESHOLD,
    fade_mode=FADE_MODE,
):
    if not os.path.exists(input_root):
        raise SystemExit(f"Input root không tồn tại: {input_root}")

    os.makedirs(output_root, exist_ok=True)

    if selected_folders:
        folder_paths = [os.path.join(input_root, folder_name) for folder_name in selected_folders]
    else:
        folder_paths = [
            os.path.join(input_root, folder_name)
            for folder_name in os.listdir(input_root)
            if os.path.isdir(os.path.join(input_root, folder_name))
        ]

    image_paths = []
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            print(f"Bỏ qua folder không tồn tại: {folder_path}")
            continue

        # Collect all image paths under this folder, then sample randomly
        folder_images = []
        for root, _, files in os.walk(folder_path):
            for file_name in files:
                if file_name.lower().endswith(IMAGE_EXTENSIONS):
                    folder_images.append(os.path.join(root, file_name))

        if not folder_images:
            continue

        # Choose up to MAX_IMAGES_PER_FOLDER images randomly from the folder
        k = min(len(folder_images), MAX_IMAGES_PER_FOLDER)
        try:
            sampled = random.sample(folder_images, k)
        except ValueError:
            # fallback: shuffle and take first k
            random.shuffle(folder_images)
            sampled = folder_images[:k]

        for img in sampled:
            image_paths.append((folder_path, img))

    total = len(image_paths)
    if total == 0:
        print(f"Không tìm thấy ảnh hợp lệ trong các folder đã chọn ở: {input_root}")
        return

    processed = 0
    skipped_read_fail = 0

    for folder_path, img_path in tqdm(image_paths, desc="Degrading images", total=total):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            skipped_read_fail += 1
            continue

        degraded = degrade_sketch_grayscale_square(
            image,
            erase_prob=erase_prob,
            min_patch=min_patch,
            max_patch=max_patch,
            edge_threshold=edge_threshold,
            fade_mode=fade_mode,
        )

        relative_path = os.path.relpath(img_path, folder_path)
        output_name = _build_output_name(relative_path)
        save_path = _get_unique_output_path(output_root, output_name)

        ok = cv2.imwrite(save_path, degraded)
        if ok:
            processed += 1

    print("===== DEGRADE FOLDER SUMMARY =====")
    print(f"Input folder        : {input_root}")
    print(f"Selected folders    : {', '.join(selected_folders) if selected_folders else 'ALL'}")
    print(f"Output folder       : {output_root}")
    print(f"Total images        : {total}")
    print(f"Processed images    : {processed}")
    print(f"Skipped read failed : {skipped_read_fail}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Degrade ảnh trong các folder con của một root folder và lưu vào một folder tổng hợp")
    parser.add_argument("--input_root", type=str, default=INPUT_ROOT, help="Root folder chứa các folder con ảnh")
    parser.add_argument("--output_root", type=str, default=OUTPUT_ROOT, help="Folder đầu ra tổng hợp")
    parser.add_argument("--erase_prob", type=float, default=ERASE_PROB, help="Tỉ lệ pixel nét được chọn")
    parser.add_argument("--min_patch", type=int, default=MIN_PATCH, help="Kích thước patch nhỏ nhất")
    parser.add_argument("--max_patch", type=int, default=MAX_PATCH, help="Kích thước patch lớn nhất")
    parser.add_argument("--edge_threshold", type=int, default=EDGE_THRESHOLD, help="Ngưỡng pixel được xem là nét")
    parser.add_argument("--fade_mode", action="store_true", help="Bật chế độ làm mờ thay vì xóa trắng")

    args = parser.parse_args()

    process_folder(
        args.input_root,
        args.output_root,
        selected_folders=SELECTED_FOLDERS,
        erase_prob=args.erase_prob,
        min_patch=args.min_patch,
        max_patch=args.max_patch,
        edge_threshold=args.edge_threshold,
        fade_mode=args.fade_mode,
    )
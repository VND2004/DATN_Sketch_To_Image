import os
import json
import shutil
import logging
import cv2
import numpy as np
import re
from tqdm import tqdm

# Nếu có RLE
try:
    from pycocotools import mask as maskUtils
    HAS_COCO = True
except:
    HAS_COCO = False

# ===== CONFIG =====
IMAGE_DIR = r"D:\THO\Bach_Khoa\Thesis\Data\train2020\train"
ANNOTATION_PATH = r"D:\THO\Bach_Khoa\Thesis\Data\instances_attributes_train2020.json"
OUTPUT_DIR = r"D:\THO\Bach_Khoa\Thesis\Data\train2020\train_no_bg_blur"

MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
MASK_BLUR_DIR = os.path.join(OUTPUT_DIR, "masks_blur")
SAVE_INTERMEDIATE_MASKS = True

BLUR_KERNEL = 11
BLUR_ITERATIONS = 3
# ==================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def load_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def build_image_dict(images):
    return {img["id"]: img["file_name"] for img in images}


def build_category_dict(categories):
    return {cat["id"]: cat["name"] for cat in categories}


def group_annotations(annotations):
    grouped = {}
    for ann in annotations:
        grouped.setdefault(ann["image_id"], []).append(ann)
    return grouped


def polygon_to_mask(image_shape, segmentation):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for seg in segmentation:
        poly = np.array(seg).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [poly], 255)
    return mask


def rle_to_mask(segmentation):
    if not HAS_COCO:
        raise ImportError("Please install pycocotools")
    return maskUtils.decode(segmentation)


def sanitize_name(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


def list_image_files(folder_path):
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = []
    for file_name in os.listdir(folder_path):
        ext = os.path.splitext(file_name)[1].lower()
        if ext in valid_exts:
            files.append(file_name)
    return sorted(files)


# ===== Gaussian + normalize =====
def smooth_and_normalize(mask):
    blurred = mask
    for _ in range(BLUR_ITERATIONS):
        blurred = cv2.GaussianBlur(blurred, (BLUR_KERNEL, BLUR_KERNEL), 0)

    # Normalize về [0, 1] để tạo soft mask
    normalized = blurred.astype(np.float32) / 255.0

    return blurred, normalized


# ===== Apply SOFT mask =====
def apply_mask(image, normalized_mask):
    alpha = normalized_mask[:, :, None]
    white_bg = np.full_like(image, 255, dtype=np.float32)
    result = image.astype(np.float32) * alpha + white_bg * (1.0 - alpha)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def process():
    LOGGER.info("[START] Gaussian mask smoothing (NO blur image)")
    LOGGER.info("[INPUT] IMAGE_DIR: %s", IMAGE_DIR)
    LOGGER.info("[CONFIG] SAVE_INTERMEDIATE_MASKS: %s", SAVE_INTERMEDIATE_MASKS)

    data = load_data(ANNOTATION_PATH)

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    image_dict = build_image_dict(images)
    file_to_image_id = {file_name: image_id for image_id, file_name in image_dict.items()}
    category_dict = build_category_dict(categories)
    grouped_anns = group_annotations(annotations)

    if not os.path.isdir(IMAGE_DIR):
        LOGGER.error("[ERROR] IMAGE_DIR does not exist: %s", IMAGE_DIR)
        return

    input_files = list_image_files(IMAGE_DIR)
    total_input_files = len(input_files)
    LOGGER.info("[INPUT] Found %d image files in folder", total_input_files)

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if SAVE_INTERMEDIATE_MASKS:
        os.makedirs(MASK_DIR, exist_ok=True)
        os.makedirs(MASK_BLUR_DIR, exist_ok=True)

    total_saved = 0
    images_with_annotations = 0
    images_processed_success = 0
    images_processed_no_output = 0
    skipped_not_in_json = 0
    skipped_no_annotation = 0
    skipped_unreadable = 0
    skipped_empty_crop = 0
    failed_save = 0

    for idx, file_name in enumerate(tqdm(input_files, desc="Processing folder images"), start=1):
        image_id = file_to_image_id.get(file_name)
        if image_id is None:
            skipped_not_in_json += 1
            continue

        anns = grouped_anns.get(image_id, [])
        if not anns:
            skipped_no_annotation += 1
            continue

        images_with_annotations += 1

        image_path = os.path.join(IMAGE_DIR, file_name)
        image = cv2.imread(image_path)
        if image is None:
            skipped_unreadable += 1
            continue

        image_saved_any = False

        for ann in anns:
            category_id = ann["category_id"]
            segmentation = ann["segmentation"]
            bbox = ann["bbox"]

            category_name = sanitize_name(
                category_dict.get(category_id, "unknown")
            )

            # ===== Step 1: mask =====
            if isinstance(segmentation, list):
                mask = polygon_to_mask(image.shape, segmentation)
            else:
                mask = rle_to_mask(segmentation)

            mask_name = f"{image_id}_{ann['id']}.png"

            # Save mask trung gian chỉ khi cần debug
            if SAVE_INTERMEDIATE_MASKS:
                cv2.imwrite(os.path.join(MASK_DIR, mask_name), mask)

            # ===== Step 2: blur + normalize =====
            mask_blur, mask_normalized = smooth_and_normalize(mask)

            if SAVE_INTERMEDIATE_MASKS:
                cv2.imwrite(os.path.join(MASK_BLUR_DIR, mask_name), mask_blur)

            # ===== Step 3: apply SOFT mask =====
            result = apply_mask(image, mask_normalized)

            # ===== Crop =====
            x, y, w, h = map(int, bbox)
            cropped = result[y:y+h, x:x+w]

            if cropped.size == 0:
                skipped_empty_crop += 1
                continue

            # ===== Save =====
            folder_name = f"{category_id}_{category_name}"
            category_folder = os.path.join(OUTPUT_DIR, folder_name)
            os.makedirs(category_folder, exist_ok=True)

            output_path = os.path.join(category_folder, mask_name)

            if cv2.imwrite(output_path, cropped):
                total_saved += 1
                image_saved_any = True
            else:
                failed_save += 1

        if image_saved_any:
            images_processed_success += 1
        else:
            images_processed_no_output += 1

        if idx % 50 == 0 or idx == total_input_files:
            LOGGER.info(
                "[PROGRESS] %d/%d images scanned | with_ann=%d | success=%d | no_output=%d | saved_crops=%d",
                idx,
                total_input_files,
                images_with_annotations,
                images_processed_success,
                images_processed_no_output,
                total_saved,
            )

    LOGGER.info("[SUMMARY] Input images in folder: %d", total_input_files)
    LOGGER.info("[SUMMARY] Images found in JSON: %d", total_input_files - skipped_not_in_json)
    LOGGER.info("[SUMMARY] Images with annotations: %d", images_with_annotations)
    LOGGER.info("[SUMMARY] Images processed successfully: %d/%d", images_processed_success, total_input_files)
    LOGGER.info("[SUMMARY] Images processed but no output: %d", images_processed_no_output)
    LOGGER.info("[SUMMARY] Skipped - not in JSON: %d", skipped_not_in_json)
    LOGGER.info("[SUMMARY] Skipped - no annotation: %d", skipped_no_annotation)
    LOGGER.info("[SUMMARY] Skipped - unreadable image: %d", skipped_unreadable)
    LOGGER.info("[SUMMARY] Skipped - empty crop: %d", skipped_empty_crop)
    LOGGER.info("[SUMMARY] Failed saves: %d", failed_save)
    LOGGER.info("[DONE] Total saved crops: %d", total_saved)


if __name__ == "__main__":
    process()
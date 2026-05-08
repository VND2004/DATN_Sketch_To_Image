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
OUTPUT_DIR = r"D:\THO\Bach_Khoa\Thesis\Data\train2020\train_no_bg"
# ==================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def build_image_dict(images):
    return {img["id"]: img["file_name"] for img in images}


def build_category_dict(categories):
    return {cat["id"]: cat["name"] for cat in categories}


def group_annotations(annotations):
    grouped = {}
    for ann in annotations:
        image_id = ann["image_id"]
        grouped.setdefault(image_id, []).append(ann)
    return grouped


def polygon_to_mask(image_shape, segmentation):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for seg in segmentation:
        poly = np.array(seg).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [poly], 255)

    return mask


def rle_to_mask(segmentation):
    if not HAS_COCO:
        raise ImportError("Please install pycocotools for RLE support")

    return maskUtils.decode(segmentation)


def sanitize_name(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


def process():
    LOGGER.info("[START] remove_background pipeline")
    LOGGER.info("Config | IMAGE_DIR=%s", IMAGE_DIR)
    LOGGER.info("Config | ANNOTATION_PATH=%s", ANNOTATION_PATH)
    LOGGER.info("Config | OUTPUT_DIR=%s", OUTPUT_DIR)
    LOGGER.info("RLE support (pycocotools): %s", "ON" if HAS_COCO else "OFF")

    LOGGER.info("[1/6] Loading annotation JSON...")
    data = load_data(ANNOTATION_PATH)

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    LOGGER.info("Loaded %d images, %d annotations, %d categories",
                len(images), len(annotations), len(categories))

    LOGGER.info("[2/6] Building lookup structures...")
    image_dict = build_image_dict(images)
    category_dict = build_category_dict(categories)
    grouped_anns = group_annotations(annotations)

    LOGGER.info("Built image_dict=%d, grouped image_ids=%d, category_dict=%d",
                len(image_dict), len(grouped_anns), len(category_dict))

    LOGGER.info("[3/6] Resetting output directory...")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        LOGGER.info("Removed existing output directory: %s", OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    LOGGER.info("Created fresh output directory: %s", OUTPUT_DIR)

    total_images = len(grouped_anns)
    total_saved = 0
    skipped_no_name = 0
    skipped_missing_file = 0
    skipped_imread_fail = 0

    LOGGER.info("[4/6] Start processing %d grouped images...", total_images)

    for image_index, image_id in enumerate(
        tqdm(grouped_anns.keys(), desc="Processing images"), start=1
    ):
        file_name = image_dict.get(image_id)
        if file_name is None:
            skipped_no_name += 1
            continue

        image_path = os.path.join(IMAGE_DIR, file_name)
        if not os.path.exists(image_path):
            skipped_missing_file += 1
            continue

        image = cv2.imread(image_path)
        if image is None:
            skipped_imread_fail += 1
            continue

        LOGGER.info("[Image %d/%d] image_id=%s | file=%s",
                    image_index, total_images, image_id, file_name)

        anns = grouped_anns[image_id]
        LOGGER.info("Found %d annotations for image_id=%s", len(anns), image_id)

        for ann_index, ann in enumerate(anns, start=1):
            category_id = ann["category_id"]
            segmentation = ann["segmentation"]
            bbox = ann["bbox"]

            category_name = category_dict.get(category_id, "unknown")
            category_name = sanitize_name(category_name)

            LOGGER.info(
                "  [Ann %d/%d] ann_id=%s | category=%s (%s) | bbox=%s",
                ann_index,
                len(anns),
                ann.get("id"),
                category_id,
                category_name,
                bbox,
            )

            # ===== Create mask =====
            if isinstance(segmentation, list):
                mask = polygon_to_mask(image.shape, segmentation)
            else:
                mask = rle_to_mask(segmentation)

            # ===== Apply mask =====
            white_bg = np.ones_like(image) * 255
            result = np.where(mask[:, :, None] > 0, image, white_bg)

            # ===== Crop =====
            x, y, w, h = map(int, bbox)
            cropped = result[y:y+h, x:x+w]

            if cropped.size == 0:
                LOGGER.warning(
                    "Empty crop for ann_id=%s (bbox=%s) on image_id=%s",
                    ann.get("id"),
                    bbox,
                    image_id,
                )
                continue

            # ===== Save =====
            folder_name = f"{category_id}_{category_name}"
            category_folder = os.path.join(OUTPUT_DIR, folder_name)
            os.makedirs(category_folder, exist_ok=True)

            output_name = f"{image_id}_{ann['id']}.png"
            output_path = os.path.join(category_folder, output_name)

            ok = cv2.imwrite(output_path, cropped)
            if ok:
                total_saved += 1
            else:
                LOGGER.warning("Failed to save: %s", output_path)

    LOGGER.info("[5/6] Processing completed")
    LOGGER.info(
        "Summary | saved=%d | skipped_no_name=%d | skipped_missing_file=%d | skipped_imread_fail=%d",
        total_saved,
        skipped_no_name,
        skipped_missing_file,
        skipped_imread_fail,
    )
    LOGGER.info("[6/6] End pipeline")


if __name__ == "__main__":
    process()
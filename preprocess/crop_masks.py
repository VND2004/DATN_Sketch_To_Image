import os
import json
import logging
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

# ===== CONFIG =====
# Chọn dataset: 'train', 'val', hoặc 'both'
DATASET_MODE = 'train'

# Train dataset
TRAIN_IMAGE_DIR = r"D:\THO\Bach_Khoa\Thesis\Data\train2020\train"
TRAIN_MASK_DIR = r"D:\THO\Bach_Khoa\Thesis\Data\train2020\(0)_masks_blur"
TRAIN_ANNOTATION_PATH = r"D:\THO\Bach_Khoa\Thesis\Data\instances_attributes_train2020.json"
TRAIN_OUTPUT_DIR = r"D:\THO\Bach_Khoa\Thesis\Data\train2020\(0)_masks_cropped"

# Validation dataset
VAL_IMAGE_DIR = r"D:\THO\Bach_Khoa\Thesis\Data\val_test2020\test"
VAL_MASK_DIR = r"D:\THO\Bach_Khoa\Thesis\Data\val_test2020\(0)_masks"
VAL_ANNOTATION_PATH = r"D:\THO\Bach_Khoa\Thesis\Data\instances_attributes_val2020.json"
VAL_OUTPUT_DIR = r"D:\THO\Bach_Khoa\Thesis\Data\val_test2020\(0)_masks_cropped"

# ==================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def load_data(json_path):
    """Load JSON annotation data"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def build_image_dict(images):
    """Build mapping from image_id to file_name"""
    return {img["id"]: img["file_name"] for img in images}


def build_category_dict(categories):
    """Build mapping from category_id to category_name"""
    return {cat["id"]: cat["name"] for cat in categories}


def group_annotations(annotations):
    """Group annotations by image_id"""
    grouped = {}
    for ann in annotations:
        image_id = ann["image_id"]
        grouped.setdefault(image_id, []).append(ann)
    return grouped


def sanitize_name(name):
    """Sanitize name for folder creation"""
    invalid_chars = r'<>:"/\|?*'
    for char in invalid_chars:
        name = name.replace(char, "_")
    return name.strip()


def process_dataset(image_dir, mask_dir, annotation_path, output_dir, dataset_name):
    """
    Process a dataset (train or val)
    
    Args:
        image_dir: Path to original images directory
        mask_dir: Path to mask images directory (original size)
        annotation_path: Path to JSON annotations
        output_dir: Path to output directory for cropped masks
        dataset_name: Name of dataset (for logging)
    """
    LOGGER.info(f"[{dataset_name}] [1/5] Loading annotations from {annotation_path}")
    
    # Load annotations
    data = load_data(annotation_path)
    image_dict = build_image_dict(data["images"])
    category_dict = build_category_dict(data["categories"])
    grouped_anns = group_annotations(data["annotations"])
    
    LOGGER.info(f"[{dataset_name}] [2/5] Loaded {len(image_dict)} images and {len(grouped_anns)} image annotations")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    total_saved = 0
    skipped_no_name = 0
    skipped_missing_file = 0
    skipped_imread_fail = 0
    
    LOGGER.info(f"[{dataset_name}] [3/5] Processing annotations")
    
    # Sort by image_id for consistent processing
    image_ids = sorted(grouped_anns.keys())
    total_images = len(image_ids)
    
    for image_index, image_id in tqdm(enumerate(image_ids, start=1), total=total_images, desc=f"[{dataset_name}] Processing images"):
        # Mask file is named: {image_id}_{ann_id}.png
        # We'll construct it during annotation loop
        
        if image_id not in grouped_anns:
            skipped_no_name += 1
            continue
        
        anns = grouped_anns[image_id]
        LOGGER.info(f"[{dataset_name}] [{image_index}/{total_images}] image_id={image_id} | Found {len(anns)} annotations")
        
        for ann_index, ann in enumerate(anns, start=1):
            category_id = ann["category_id"]
            bbox = ann.get("bbox")
            ann_id = ann['id']
            
            if bbox is None:
                LOGGER.warning(f"[{dataset_name}] No bbox for ann_id={ann_id}")
                continue
            
            # Mask file is named: {image_id}_{ann_id}.png
            mask_path = os.path.join(mask_dir, f"{image_id}_{ann_id}.png")
            
            # Mask file is named: {image_id}_{ann_id}.png
            mask_path = os.path.join(mask_dir, f"{image_id}_{ann_id}.png")
            
            # Check if mask file exists
            if not os.path.isfile(mask_path):
                LOGGER.warning(f"[{dataset_name}] Mask file not found: {mask_path}")
                skipped_missing_file += 1
                continue
            
            # Read mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                LOGGER.warning(f"[{dataset_name}] Failed to read mask: {mask_path}")
                skipped_imread_fail += 1
                continue
            
            category_name = category_dict.get(category_id, "unknown")
            category_name = sanitize_name(category_name)
            
            LOGGER.info(
                f"[{dataset_name}]   [Ann {ann_index}/{len(anns)}] ann_id={ann_id} | "
                f"category={category_id} ({category_name}) | bbox={bbox}"
            )
            
            # ===== Crop mask =====
            x, y, w, h = map(int, bbox)
            cropped_mask = mask[y:y+h, x:x+w]
            
            if cropped_mask.size == 0:
                LOGGER.warning(
                    f"[{dataset_name}] Empty crop for ann_id={ann_id} (bbox={bbox}) on image_id={image_id}"
                )
                continue
            
            # ===== Save =====
            os.makedirs(output_dir, exist_ok=True)
            
            output_name = f"{image_id}_{ann_id}.png"
            output_path = os.path.join(output_dir, output_name)
            
            ok = cv2.imwrite(output_path, cropped_mask)
            if ok:
                total_saved += 1
            else:
                LOGGER.warning(f"[{dataset_name}] Failed to save: {output_path}")
    
    LOGGER.info(f"[{dataset_name}] [4/5] Processing completed")
    LOGGER.info(
        f"[{dataset_name}] Summary | saved={total_saved} | skipped_no_name={skipped_no_name} | "
        f"skipped_missing_file={skipped_missing_file} | skipped_imread_fail={skipped_imread_fail}"
    )
    
    return total_saved


def main():
    """Main pipeline"""
    LOGGER.info("=" * 60)
    LOGGER.info("CROP MASKS PIPELINE")
    LOGGER.info(f"MODE: {DATASET_MODE.upper()}")
    LOGGER.info("=" * 60)
    
    total_all = 0
    
    # Process train dataset
    if DATASET_MODE in ['train', 'both']:
        if os.path.isdir(TRAIN_MASK_DIR) and os.path.isfile(TRAIN_ANNOTATION_PATH):
            LOGGER.info("\n" + "=" * 60)
            LOGGER.info("PROCESSING TRAIN DATASET")
            LOGGER.info("=" * 60)
            total_train = process_dataset(
                TRAIN_IMAGE_DIR,
                TRAIN_MASK_DIR,
                TRAIN_ANNOTATION_PATH,
                TRAIN_OUTPUT_DIR,
                "TRAIN"
            )
            total_all += total_train
            LOGGER.info(f"Train output saved to: {TRAIN_OUTPUT_DIR}\n")
        else:
            LOGGER.warning("Train dataset paths not found, skipping train processing")
    
    # Process val/test dataset
    if DATASET_MODE in ['val', 'both']:
        if os.path.isdir(VAL_MASK_DIR) and os.path.isfile(VAL_ANNOTATION_PATH):
            LOGGER.info("\n" + "=" * 60)
            LOGGER.info("PROCESSING VAL/TEST DATASET")
            LOGGER.info("=" * 60)
            total_val = process_dataset(
                VAL_IMAGE_DIR,
                VAL_MASK_DIR,
                VAL_ANNOTATION_PATH,
                VAL_OUTPUT_DIR,
                "VAL"
            )
            total_all += total_val
            LOGGER.info(f"Val output saved to: {VAL_OUTPUT_DIR}\n")
        else:
            LOGGER.warning("Val dataset paths not found, skipping val processing")
    
    LOGGER.info("=" * 60)
    LOGGER.info(f"PIPELINE COMPLETE - Total masks cropped: {total_all}")
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()

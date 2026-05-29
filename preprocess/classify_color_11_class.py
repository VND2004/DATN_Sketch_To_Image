import cv2
import numpy as np
import os
import shutil
import argparse
import json
from collections import Counter

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# =========================================================
# 1. PATH CONFIG
# =========================================================
# INPUT_ROOT = r'D:\THO\Bach_Khoa\Thesis\Data\train2020\(3)_filtered_by_category'
# MASK_FOLDER = r'D:\THO\Bach_Khoa\Thesis\Data\train2020\(0)_masks_cropped'
# OUTPUT_FOLDER = r'D:\THO\Bach_Khoa\Thesis\Data\train2020\(5)_color_classified'

INPUT_ROOT = r'D:\Git\DATN_Sketch_To_Image\Data\train2020\(3)_filtered_by_category'
MASK_FOLDER = r'D:\Git\DATN_Sketch_To_Image\Data\train2020\(0)_masks_cropped'
OUTPUT_FOLDER = r'D:\Git\DATN_Sketch_To_Image\Data\train2020\(5)_color_classified_11_classes'

# =========================================================
# 2. COLOR CONFIG
# OpenCV HSV:
# H: 0-179
# S: 0-255
# V: 0-255
# =========================================================

CONFIG = {

    # -------------------------
    # Neutral colors
    # -------------------------
    'BLACK_V_MAX': 35,
    # Soft-black catches dark, low-saturation pixels that are visually black.
    'BLACK_V_SOFT_MAX': 55,
    'BLACK_S_MAX': 45,

    'WHITE_S_MAX': 30,
    'WHITE_V_MIN': 200,

    'GRAY_S_MAX': 40,
    'GRAY_V_MIN': 56,
    'GRAY_V_MAX': 199,

    # Unknown pixels darker than this will be considered black instead of gray.
    'BLACK_UNKNOWN_V_MAX': 55,

    # -------------------------
    # Brown
    # Brown = dark orange
    # -------------------------
    'BROWN_H_MIN': 5,
    'BROWN_H_MAX': 20,
    'BROWN_S_MIN': 60,
    'BROWN_V_MAX': 170,

    # -------------------------
    # Main hue ranges
    # -------------------------
    'RED_RANGES': [
        (0, 5),
        (170, 179)
    ],

    'ORANGE_RANGE': (6, 19),

    'YELLOW_RANGE': (20, 38),

    'GREEN_RANGE': (33, 92),

    'BLUE_RANGE': (93, 125),

    'PURPLE_RANGE': (126, 158),

    'PINK_RANGE': (154, 169),

    # Allow slightly lower saturation for pink (pastel pink)
    'PINK_S_MIN': 30,

    # saturation threshold
    'COLOR_S_MIN': 40
}


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================

def in_ranges(h, ranges):
    for low, high in ranges:
        if low <= h <= high:
            return True
    return False


def classify_pixels(hsv_pixels):
    """
    Classify each pixel into:
    black
    white
    gray
    red
    orange
    yellow
    green
    blue
    purple
    pink
    brown
    """

    H = hsv_pixels[:, 0]
    S = hsv_pixels[:, 1]
    V = hsv_pixels[:, 2]

    labels = np.full(len(hsv_pixels), 'unknown', dtype=object)

    # =====================================================
    # 1. BLACK
    # =====================================================
    black_mask = (
        (V <= CONFIG['BLACK_V_MAX']) |
        ((V <= CONFIG['BLACK_V_SOFT_MAX']) & (S <= CONFIG['BLACK_S_MAX']))
    )
    labels[black_mask] = 'black'

    # =====================================================
    # 2. WHITE
    # =====================================================
    white_mask = (
        (labels == 'unknown') &
        (S <= CONFIG['WHITE_S_MAX']) &
        (V >= CONFIG['WHITE_V_MIN'])
    )
    labels[white_mask] = 'white'

    # =====================================================
    # 3. GRAY
    # =====================================================
    gray_mask = (
        (labels == 'unknown') &
        (S <= CONFIG['GRAY_S_MAX']) &
        (V >= CONFIG['GRAY_V_MIN']) &
        (V <= CONFIG['GRAY_V_MAX'])
    )
    labels[gray_mask] = 'gray'

    # =====================================================
    # 4. BROWN
    # Brown must come BEFORE orange
    # =====================================================
    brown_mask = (
        (labels == 'unknown') &
        (H >= CONFIG['BROWN_H_MIN']) &
        (H <= CONFIG['BROWN_H_MAX']) &
        (S >= CONFIG['BROWN_S_MIN']) &
        (V <= CONFIG['BROWN_V_MAX'])
    )
    labels[brown_mask] = 'brown'

    # =====================================================
    # 5. Remaining colorful pixels
    # =====================================================
    colorful_mask = (
        (labels == 'unknown') &
        (S >= CONFIG['COLOR_S_MIN'])
    )

    # RED
    red_mask = colorful_mask & (
        ((H >= 0) & (H <= 5)) |
        ((H >= 170) & (H <= 179))
    )
    labels[red_mask] = 'red'

    # ORANGE
    orange_mask = colorful_mask & (
        (H >= CONFIG['ORANGE_RANGE'][0]) &
        (H <= CONFIG['ORANGE_RANGE'][1])
    )
    labels[orange_mask] = 'orange'

    # YELLOW
    yellow_mask = colorful_mask & (
        (H >= CONFIG['YELLOW_RANGE'][0]) &
        (H <= CONFIG['YELLOW_RANGE'][1])
    )
    labels[yellow_mask] = 'yellow'

    # GREEN
    green_mask = colorful_mask & (
        (H >= CONFIG['GREEN_RANGE'][0]) &
        (H <= CONFIG['GREEN_RANGE'][1])
    )
    labels[green_mask] = 'green'

    # BLUE
    blue_mask = colorful_mask & (
        (H >= CONFIG['BLUE_RANGE'][0]) &
        (H <= CONFIG['BLUE_RANGE'][1])
    )
    labels[blue_mask] = 'blue'

    # PINK
    pink_mask = (
        (labels == 'unknown') &
        (H >= CONFIG['PINK_RANGE'][0]) &
        (H <= CONFIG['PINK_RANGE'][1]) &
        (S >= CONFIG['PINK_S_MIN'])
    )
    labels[pink_mask] = 'pink'

    # PURPLE
    purple_mask = colorful_mask & (
        (H >= CONFIG['PURPLE_RANGE'][0]) &
        (H <= CONFIG['PURPLE_RANGE'][1])
    )
    labels[purple_mask] = 'purple'

    # =====================================================
    # Remaining unknown pixels:
    # - very dark -> black
    # - otherwise -> gray
    # =====================================================
    unknown_mask = (labels == 'unknown')
    labels[unknown_mask & (V <= CONFIG['BLACK_UNKNOWN_V_MAX'])] = 'black'
    labels[labels == 'unknown'] = 'gray'

    return labels


# =========================================================
# 4. IMAGE CLASSIFICATION
# =========================================================

def classify_color(image, mask):

    # lấy pixel trong vùng mask
    pixels = image[mask > 128]

    if len(pixels) == 0:
        return 'white', {}

    hsv_pixels = cv2.cvtColor(
        pixels.reshape(-1, 1, 3),
        cv2.COLOR_BGR2HSV
    ).reshape(-1, 3)

    # classify each pixel
    pixel_labels = classify_pixels(hsv_pixels)

    # count
    unique, counts = np.unique(pixel_labels, return_counts=True)

    color_count = {
        k: int(v)
        for k, v in zip(unique, counts)
    }

    # dominant label
    dominant_label = unique[np.argmax(counts)]

    return dominant_label, color_count


# =========================================================
# 5. MAIN PROCESS
# =========================================================

def process_input_root(
    input_root,
    mask_folder,
    output_root,
    output_json_path
):

    categories = [
        'black',
        'white',
        'gray',
        'red',
        'green',
        'yellow',
        'blue',
        'brown',
        'purple',
        'pink',
        'orange'
    ]

    # clean output
    if os.path.exists(output_root):
        shutil.rmtree(output_root)

    subfolders = [
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    ]

    total_images = 0

    for sub in subfolders:
        subpath = os.path.join(input_root, sub)

        total_images += len([
            f for f in os.listdir(subpath)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    print(f"Processing {total_images} images...")

    processed = 0
    results = []
    label_counter = Counter()

    for sub in tqdm(subfolders, desc="Folders"):

        subpath = os.path.join(input_root, sub)
        out_subroot = os.path.join(output_root, sub)

        # create category folders
        for cat in categories:
            os.makedirs(
                os.path.join(out_subroot, cat),
                exist_ok=True
            )

        image_files = [
            f for f in os.listdir(subpath)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        for filename in tqdm(
            image_files,
            desc=f"{sub}",
            leave=False
        ):

            img_path = os.path.join(subpath, filename)
            mask_path = os.path.join(mask_folder, filename)

            if not os.path.exists(mask_path):
                continue

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                continue

            # resize mask if needed
            if img.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            # classify
            label, color_count = classify_color(img, mask)
            label_counter[label] += 1

            # rename file
            name, ext = os.path.splitext(filename)

            count_str = "_".join([
                f"{k}{v}"
                for k, v in sorted(color_count.items())
            ])

            new_filename = f"{name}__{label}__{count_str}{ext}"

            # copy
            dst = os.path.join(
                out_subroot,
                label,
                new_filename
            )

            shutil.copy(img_path, dst)

            results.append({
                'subfolder': sub,
                'filename': filename,
                'output_filename': new_filename,
                'label': label,
                'color_count': color_count,
                'source_path': os.path.relpath(img_path, input_root),
                'mask_path': os.path.relpath(mask_path, mask_folder),
                'output_path': os.path.relpath(dst, output_root),
            })

            processed += 1

    os.makedirs(
        os.path.dirname(output_json_path),
        exist_ok=True
    )

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'input_root': input_root,
            'mask_folder': mask_folder,
            'output_folder': output_root,
            'total_processed': processed,
            'label_summary': dict(sorted(label_counter.items())),
            'items': results,
        }, f, ensure_ascii=False, indent=2)

    print(f"Done! Processed {processed} images.")
    print("Label summary:")
    for label in categories:
        print(f"  {label}: {label_counter.get(label, 0)}")
    print(f"JSON saved to: {output_json_path}")


# =========================================================
# 6. ENTRY
# =========================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Color classification using masked pixels'
    )

    parser.add_argument(
        '--input_root',
        type=str,
        default=INPUT_ROOT
    )

    parser.add_argument(
        '--mask_folder',
        type=str,
        default=MASK_FOLDER
    )

    parser.add_argument(
        '--output_folder',
        type=str,
        default=OUTPUT_FOLDER
    )

    parser.add_argument(
        '--output_json',
        type=str,
        default=None
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_root):
        raise SystemExit(
            f"Input root does not exist: {args.input_root}"
        )

    if not os.path.exists(args.mask_folder):
        raise SystemExit(
            f"Mask folder does not exist: {args.mask_folder}"
        )

    output_json_path = (
        args.output_json
        or os.path.join(
            args.output_folder,
            'color_info.json'
        )
    )

    process_input_root(
        args.input_root,
        args.mask_folder,
        args.output_folder,
        output_json_path
    )
import cv2
import numpy as np
import random


def degrade_sketch_grayscale_square(
    image,
    erase_prob=0.02,
    min_patch=3,
    max_patch=10,
    edge_threshold=220,
    fade_mode=True
):
    """
    Degrade sketch grayscale bằng cách làm mất nét
    với patch hình vuông.

    Args:
        image            : grayscale sketch
        erase_prob       : tỉ lệ pixel nét được chọn
        min_patch        : kích thước patch nhỏ nhất
        max_patch        : kích thước patch lớn nhất
        edge_threshold   : pixel < threshold được xem là nét
        fade_mode        :
            True  -> fade nét
            False -> xóa trắng hoàn toàn

    Returns:
        degraded grayscale sketch
    """

    degraded = image.copy().astype(np.float32)

    h, w = degraded.shape

    # =====================================================
    # Chỉ chọn vùng có nét
    # =====================================================
    edge_pixels = np.argwhere(degraded < edge_threshold)

    num_pixels = len(edge_pixels)

    if num_pixels == 0:
        return image

    num_erase = int(num_pixels * erase_prob)

    selected_indices = np.random.choice(
        num_pixels,
        num_erase,
        replace=False
    )

    for idx in selected_indices:

        y, x = edge_pixels[idx]

        # random kích thước patch
        patch_size = random.randint(
            min_patch,
            max_patch
        )

        half = patch_size // 2

        x1 = max(0, x - half)
        y1 = max(0, y - half)

        x2 = min(w, x + half)
        y2 = min(h, y + half)

        # =================================================
        # Fade nét
        # =================================================
        if fade_mode:

            alpha = random.uniform(0.4, 0.9)

            degraded[y1:y2, x1:x2] = (
                degraded[y1:y2, x1:x2] * (1 - alpha)
                + 255 * alpha
            )

        else:
            # xóa trắng hoàn toàn
            degraded[y1:y2, x1:x2] = 255

    degraded = np.clip(
        degraded,
        0,
        255
    ).astype(np.uint8)

    return degraded


# =========================================================
# Example
# =========================================================

if __name__ == "__main__":

    img = cv2.imread(
        r"D:\Git\DATN_Sketch_To_Image\Data\val_test2020\(4)_sketch_canny\1_top__t_shirt__sweatshirt\20490_5010.png",
        cv2.IMREAD_GRAYSCALE
    )

    degraded = degrade_sketch_grayscale_square(
        img,
        erase_prob=0.003,
        min_patch=10,
        max_patch=30,
        edge_threshold=230,
        fade_mode=False
    )

    cv2.imwrite(
        "degraded_square.png",
        degraded
    )
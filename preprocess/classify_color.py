import cv2
import numpy as np
import os
import shutil
import argparse
import json
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# =========================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# INPUT_ROOT: folder cha chứa nhiều folder con (mỗi folder con chứa nhiều ảnh)
# MASK_FOLDER: folder cố định chứa các mask (tên mask khớp tên ảnh)
# OUTPUT_FOLDER: folder đầu ra sẽ chứa các folder con cùng tên với input
# =========================================================
INPUT_ROOT = r'D:\THO\Bach_Khoa\Thesis\Data\train2020\(3)_filtered_by_category'
MASK_FOLDER = r'D:\THO\Bach_Khoa\Thesis\Data\train2020\(0)_masks_cropped'
OUTPUT_FOLDER = r'D:\THO\Bach_Khoa\Thesis\Data\train2020\(5)_color_classified'

# =========================================================
# 2. CẤU HÌNH DẢI MÀU (TỰ ĐIỀU CHỈNH TẠI ĐÂY)
# =========================================================
CONFIG = {
    'BLACK_V_MAX': 55,       # V < 55 là Đen (Tăng nếu muốn nhận thêm đồ xám đậm vào nhóm Đen)
    'WHITE_S_MAX': 30,       # S < 30 là nhóm không màu (GIẢM số này nếu màu nhạt đang bị nhận là trắng)
    'WHITE_V_MIN': 160,      # V >= 160 và S thấp thì là Trắng (TĂNG số này nếu màu xám đang bị nhận là trắng)
    
    # H: 0-25 (Đỏ, Cam, Vàng) và 155-180 (Hồng, Đỏ đô)
    'WARM_H_RANGES': [(0, 25), (155, 180)] 
}

# =========================================================
# 3. HÀM XỬ LÝ CHÍNH
# =========================================================

def is_warm(h_value):
    """Kiểm tra xem giá trị Hue có thuộc gam nóng không"""
    for (low, high) in CONFIG['WARM_H_RANGES']:
        if low <= h_value <= high:
            return True
    return False

def classify_color(image, mask):
    # Lấy các pixel trong vùng mask
    pixels = image[mask > 128]
    if len(pixels) == 0:
        return 'White', None, None, None

    # Chuyển sang HSV và tính Median để tránh nhiễu
    pixels_hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    avg_h = np.median(pixels_hsv[:, 0])
    avg_s = np.median(pixels_hsv[:, 1])
    avg_v = np.median(pixels_hsv[:, 2])

    # --- LOGIC PHÂN LOẠI ---
    
    # 1. Ưu tiên kiểm tra màu Đen trước
    if avg_v < CONFIG['BLACK_V_MAX']:
        return 'Black', avg_h, avg_s, avg_v
    
    # 2. Kiểm tra màu Trắng (Dựa trên độ bão hòa thấp và độ sáng cao)
    if avg_s < CONFIG['WHITE_S_MAX'] and avg_v > CONFIG['WHITE_V_MIN']:
        return 'White', avg_h, avg_s, avg_v
    
    # 3. Nếu không phải đen/trắng thì xét đến màu sắc (Nóng/Lạnh)
    if is_warm(avg_h):
        return 'Warm', avg_h, avg_s, avg_v
    else:
        return 'Cold', avg_h, avg_s, avg_v

# =========================================================
# 4. THỰC THI
# =========================================================

def process_input_root(input_root, mask_folder, output_root, output_json_path):
    categories = ['Black', 'White', 'Warm', 'Cold']

    # Xóa folder output cũ nếu tồn tại để đảm bảo bắt đầu sạch
    if os.path.exists(output_root):
        shutil.rmtree(output_root)

    # Duyệt các folder con trong input_root
    subfolders = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    total_images = 0
    # Đếm tổng số ảnh (chỉ để hiển thị)
    for sub in subfolders:
        subpath = os.path.join(input_root, sub)
        total_images += len([f for f in os.listdir(subpath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"Đang xử lý {total_images} ảnh trong {len(subfolders)} folder con...")

    processed = 0
    results = []
    for sub in tqdm(subfolders, desc="Processing folders", total=len(subfolders)):
        subpath = os.path.join(input_root, sub)
        out_subroot = os.path.join(output_root, sub)
        # Tạo folder output cho folder con và các category bên trong
        for cat in categories:
            os.makedirs(os.path.join(out_subroot, cat), exist_ok=True)

        image_files = [f for f in os.listdir(subpath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for filename in tqdm(image_files, desc=f"Processing {sub}", leave=False):
            img_path = os.path.join(subpath, filename)
            mask_path = os.path.join(mask_folder, filename)

            # Chỉ xử lý nếu có file mask tương ứng
            if not os.path.exists(mask_path):
                continue

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                continue

            # Đảm bảo size mask khớp với ảnh
            if img.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

            # Phân loại (trả về label và chỉ số HSV)
            label, avg_h, avg_s, avg_v = classify_color(img, mask)

            # Chuẩn bị tên file mới có kèm chỉ số HSV
            name, ext = os.path.splitext(filename)
            if avg_h is None:
                h_str, s_str, v_str = 'NA', 'NA', 'NA'
            else:
                h_str = str(int(round(avg_h)))
                s_str = str(int(round(avg_s)))
                v_str = str(int(round(avg_v)))

            new_filename = f"{name}__H{h_str}_S{s_str}_V{v_str}{ext}"

            # Copy file vào folder output tương ứng với folder con và label
            dst = os.path.join(out_subroot, label, new_filename)
            shutil.copy(img_path, dst)
            results.append({
                'subfolder': sub,
                'filename': filename,
                'output_filename': new_filename,
                'label': label,
                'h': None if avg_h is None else int(round(avg_h)),
                's': None if avg_s is None else int(round(avg_s)),
                'v': None if avg_v is None else int(round(avg_v)),
                'source_path': os.path.relpath(img_path, input_root),
                'mask_path': os.path.relpath(mask_path, mask_folder),
                'output_path': os.path.relpath(dst, output_root),
            })
            processed += 1

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'input_root': input_root,
            'mask_folder': mask_folder,
            'output_folder': output_root,
            'total_processed': processed,
            'items': results,
        }, f, ensure_ascii=False, indent=2)

    print(f"Hoàn thành! Đã phân loại {processed} ảnh vào {output_root}")
    print(f"Đã lưu JSON vào {output_json_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phân loại màu cho nhiều folder ảnh, dùng mask cố định')
    parser.add_argument('--input_root', type=str, default=INPUT_ROOT,
                        help='Folder chứa nhiều folder con (mỗi folder con nhiều ảnh)')
    parser.add_argument('--mask_folder', type=str, default=MASK_FOLDER,
                        help='Folder cố định chứa các mask (tên mask khớp tên ảnh)')
    parser.add_argument('--output_folder', type=str, default=OUTPUT_FOLDER,
                        help='Folder output sẽ chứa các folder con cùng tên với input')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Đường dẫn file JSON đầu ra. Mặc định là output_folder/color_info.json')

    args = parser.parse_args()

    # Validate
    if not os.path.exists(args.input_root):
        raise SystemExit(f"Input root không tồn tại: {args.input_root}")
    if not os.path.exists(args.mask_folder):
        raise SystemExit(f"Mask folder không tồn tại: {args.mask_folder}")

    output_json_path = args.output_json or os.path.join(args.output_folder, 'color_info.json')
    process_input_root(args.input_root, args.mask_folder, args.output_folder, output_json_path)
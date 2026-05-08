import os
import shutil
from tqdm import tqdm

# ====== INPUT / OUTPUT ======
input_root = r"D:\THO\Bach_Khoa\Thesis\Data\train2020\(1)_images_filtered"
output_root = r"D:\THO\Bach_Khoa\Thesis\Data\train2020\(3)_filtered_by_category"

# ====== DANH SÁCH FOLDER CẦN LỌC ======
target_folders = {
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
    "7_shorts"
}

# ====== XÓA OUTPUT CŨ ======
if os.path.exists(output_root):
    shutil.rmtree(output_root)

os.makedirs(output_root)

# folder tổng hợp
all_images_folder = os.path.join(output_root, "all_images")
os.makedirs(all_images_folder)

# ====== THỐNG KÊ ======
stats = {}
total_images = 0

# ====== LỌC FOLDER HỢP LỆ ======
valid_folders = [
    f for f in os.listdir(input_root)
    if os.path.isdir(os.path.join(input_root, f)) and f in target_folders
]

# ====== XỬ LÝ ======
for folder_name in tqdm(valid_folders, desc="Processing folders"):
    input_subfolder = os.path.join(input_root, folder_name)
    output_subfolder = os.path.join(output_root, folder_name)

    os.makedirs(output_subfolder, exist_ok=True)

    count = 0

    image_files = [
        f for f in os.listdir(input_subfolder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ]

    for file_name in tqdm(image_files, desc=f"{folder_name}", leave=False):
        input_file = os.path.join(input_subfolder, file_name)

        # copy vào folder riêng
        shutil.copy2(input_file, os.path.join(output_subfolder, file_name))

        # copy vào folder tổng (tránh trùng tên)
        base, ext = os.path.splitext(file_name)
        new_name = file_name
        i = 1
        while os.path.exists(os.path.join(all_images_folder, new_name)):
            new_name = f"{base}_{i}{ext}"
            i += 1

        shutil.copy2(input_file, os.path.join(all_images_folder, new_name))

        count += 1
        total_images += 1

    stats[folder_name] = count

# ====== IN THỐNG KÊ ======
print("\n===== THỐNG KÊ =====")
for k, v in stats.items():
    print(f"{k}: {v} ảnh")

print(f"\nTổng số ảnh: {total_images}")
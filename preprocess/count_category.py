import os
from tqdm import tqdm

# ================== CONFIG ==================
root_folder = r"D:\THO\Bach_Khoa\Thesis\Data\train2020\(1)_images_filtered"
output_txt = r"D:\THO\Bach_Khoa\Thesis\Data\train2020\category_counts.txt"

# Các định dạng ảnh hỗ trợ
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
# ============================================

# Lấy danh sách folder con
subfolders = [f for f in os.listdir(root_folder) 
              if os.path.isdir(os.path.join(root_folder, f))]

results = []

# Duyệt từng folder con
for subfolder in tqdm(subfolders, desc="Processing folders"):
    subfolder_path = os.path.join(root_folder, subfolder)

    count = 0
    for file in os.listdir(subfolder_path):
        if file.lower().endswith(IMG_EXTENSIONS):
            count += 1

    results.append((subfolder, count))

# Sắp xếp giảm dần theo số lượng ảnh
results.sort(key=lambda x: -x[1])

# Ghi ra file txt
with open(output_txt, 'w', encoding='utf-8') as f:
    for folder_name, count in results:
        f.write(f"{folder_name}: {count}\n")

print("Done! Saved to:", output_txt)
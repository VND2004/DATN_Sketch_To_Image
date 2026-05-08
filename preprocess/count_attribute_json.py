import json
from collections import defaultdict
from tqdm import tqdm

# ================== CONFIG ==================
json_path = r"D:\THO\Bach_Khoa\Thesis\Data\instances_attributes_train2020.json"
output_txt = r"D:\THO\Bach_Khoa\Thesis\Data\train_output_stats_sort_quantity.txt"
# ============================================

# Load JSON
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

annotations = data.get("annotations", [])
categories = data.get("categories", [])
attributes = data.get("attributes", [])

# Map id -> name
cat_id2name = {c["id"]: c["name"] for c in categories}
attr_id2name = {a["id"]: a["name"] for a in attributes}

# ================== COUNT ==================

# 1. Count theo instance
category_count = defaultdict(int)
attribute_count = defaultdict(int)

# 2. Count theo unique image
category_image_map = defaultdict(set)
attribute_image_map = defaultdict(set)

for ann in tqdm(annotations, desc="Processing annotations"):
    image_id = ann["image_id"]
    cat_id = ann["category_id"]
    attr_ids = ann.get("attribute_ids", [])

    # ===== Instance count =====
    category_count[cat_id] += 1
    for attr_id in attr_ids:
        attribute_count[attr_id] += 1

    # ===== Unique image count =====
    category_image_map[cat_id].add(image_id)
    for attr_id in attr_ids:
        attribute_image_map[attr_id].add(image_id)

# Convert unique image count
category_unique_count = {k: len(v) for k, v in category_image_map.items()}
attribute_unique_count = {k: len(v) for k, v in attribute_image_map.items()}

# ================== SAVE ==================
with open(output_txt, 'w', encoding='utf-8') as f:
    f.write("===== CATEGORY (INSTANCE COUNT) =====\n")
    for cat_id, count in sorted(category_count.items(), key=lambda x: -x[1]):
        name = cat_id2name.get(cat_id, "Unknown")
        f.write(f"{cat_id} - {name}: {count}\n")

    f.write("\n===== CATEGORY (UNIQUE IMAGE COUNT) =====\n")
    for cat_id, count in sorted(category_unique_count.items(), key=lambda x: -x[1]):
        name = cat_id2name.get(cat_id, "Unknown")
        f.write(f"{cat_id} - {name}: {count}\n")

    f.write("\n===== ATTRIBUTE (INSTANCE COUNT) =====\n")
    for attr_id, count in sorted(attribute_count.items(), key=lambda x: -x[1]):
        name = attr_id2name.get(attr_id, "Unknown")
        f.write(f"{attr_id} - {name}: {count}\n")

    f.write("\n===== ATTRIBUTE (UNIQUE IMAGE COUNT) =====\n")
    for attr_id, count in sorted(attribute_unique_count.items(), key=lambda x: -x[1]):
        name = attr_id2name.get(attr_id, "Unknown")
        f.write(f"{attr_id} - {name}: {count}\n")

print("Done! Saved to:", output_txt)
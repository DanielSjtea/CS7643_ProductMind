import os
import json
from glob import glob
from tqdm import tqdm

DATA_DIR = "abo-listings"
IMAGE_DIR = "abo-images-small"
OUT_FILE = "data.jsonl"

# Print debug info
print(f"Looking for JSON files in: {os.path.abspath(DATA_DIR)}")
json_files = glob(os.path.join(DATA_DIR, "listings_*.json"))
print(f"Found {len(json_files)} JSON files")

# 打开输出文件
with open(OUT_FILE, "w", encoding="utf-8") as out_f:
    total = 0
    for json_path in json_files:
        print(f"\nProcessing file: {json_path}")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        
                        # Extract required fields
                        brand_list = item.get("brand", [])
                        brand = brand_list[0].get("value") if brand_list else None
                        
                        item_id = item.get("item_id")
                        
                        item_name_list = item.get("item_name", [])
                        title = item_name_list[0].get("value") if item_name_list else None
                        
                        description = None
                        bullet_points = item.get("bullet_point", [])
                        if bullet_points:
                            description = " ".join(point.get("value", "") for point in bullet_points)

                        if not all([item_id, title, description, brand]):
                            print(f"Skipping record due to missing required field - id:{item_id}, title:{bool(title)}, desc:{bool(description)}, brand:{bool(brand)}")
                            continue

                        # 使用任意可用的图片
                        # 从abo-images-small目录中选择一个可用的图片
                        image_dirs = [d for d in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, d))]
                        if not image_dirs:
                            print(f"Skipping record {item_id} - no image directories found")
                            continue
                            
                        # 选择第一个目录
                        folder = image_dirs[0]
                        image_files = [f for f in os.listdir(os.path.join(IMAGE_DIR, folder)) if f.endswith('.jpg')]
                        if not image_files:
                            print(f"Skipping record {item_id} - no image files found in {folder}")
                            continue
                            
                        # 选择第一个图片
                        image_file = image_files[0]
                        image_path = os.path.join(IMAGE_DIR, folder, image_file)

                        record = {
                            "item_id": item_id,
                            "image_path": image_path,
                            "title": title,
                            "description": description,
                            "brand": brand
                        }

                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total += 1
                        
                        if total % 1000 == 0:
                            print(f"Processed {total} records...")
                            
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON lines
                    except Exception as e:
                        print(f"Error processing record: {str(e)}")
                        continue
                        
        except Exception as e:
            print(f"Error processing file {json_path}: {str(e)}")

print(f"\n✅ 共写入 {total} 条记录到 {OUT_FILE}")

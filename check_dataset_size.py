import os
import pandas as pd

# Paths to dataset components
images_dir = r"C:\Users\DELL\Desktop\DualCoOp-main\MLRSNet\MLRSNet-master\Images"
labels_dir = r"C:\Users\DELL\Desktop\DualCoOp-main\MLRSNet\MLRSNet-master\Labels"
categories_file = r"C:\Users\DELL\Desktop\DualCoOp-main\MLRSNet\MLRSNet-master\Categories_names.xlsx"

# Count total images
total_images = 0
for root, dirs, files in os.walk(images_dir):
    total_images += len([f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))])

# Load categories
categories = pd.read_excel(categories_file, sheet_name="Categories")["Categories"].tolist()

print(f"Total number of images: {total_images}")
print(f"Number of categories: {len(categories)}")

# Count images per category
print("\nImages per category:")
for category in categories:
    category_path = os.path.join(images_dir, category)
    if os.path.exists(category_path):
        num_images = len([f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"{category}: {num_images}") 
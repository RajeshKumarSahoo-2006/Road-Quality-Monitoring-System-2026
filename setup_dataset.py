import os
import shutil
from pathlib import Path
import glob

# Source directories
source_dirs = ['data/input', 'Test']

# Copy images to dataset structure
all_images = []
for source_dir in source_dirs:
    if Path(source_dir).exists():
        images = glob.glob(f"{source_dir}/*.*")
        for img in images:
            if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(img)

print(f"Found {len(all_images)} images")

if len(all_images) == 0:
    print("No images found!")
    exit(1)

# Split into train/valid/test (70/15/15)
train_count = int(len(all_images) * 0.7)
valid_count = int(len(all_images) * 0.15)

# Copy images
for i, img in enumerate(all_images):
    try:
        basename = Path(img).name
        if i < train_count:
            dest = f'dataset/train/images/{basename}'
        elif i < train_count + valid_count:
            dest = f'dataset/valid/images/{basename}'
        else:
            dest = f'dataset/test/images/{basename}'
        
        shutil.copy2(img, dest)
        print(f"✓ Copied {basename}")
    except Exception as e:
        print(f"✗ Error copying {img}: {e}")

# Create sample labels (empty/zero detections for demo)
for split in ['train', 'valid', 'test']:
    images_dir = f'dataset/{split}/images'
    labels_dir = f'dataset/{split}/labels'
    
    if os.path.exists(images_dir):
        for img_file in os.listdir(images_dir):
            label_file = Path(img_file).stem + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            # Create empty label files (no detections) for demo
            with open(label_path, 'w') as f:
                f.write('')

print(f"\n✓ Images organized:")
print(f"  Train: {len(os.listdir('dataset/train/images'))} images")
print(f"  Valid: {len(os.listdir('dataset/valid/images'))} images")
print(f"  Test:  {len(os.listdir('dataset/test/images'))} images")
print("\n✓ Dataset structure ready for training!")

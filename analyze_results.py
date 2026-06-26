import pandas as pd
import os

# Load CSV
df = pd.read_csv('detections.csv')

print("\n" + "="*70)
print("TRAFFIC SIGN DETECTION RESULTS".center(70))
print("="*70)

# Basic statistics
print(f"\nTotal Detections: {len(df)}")
print(f"Unique Traffic Signs: {df['Traffic_Sign_Name'].nunique()}")
print(f"Average Confidence: {df['Confidence_Score'].mean():.4f}")
print(f"Min Confidence: {df['Confidence_Score'].min():.4f}")
print(f"Max Confidence: {df['Confidence_Score'].max():.4f}")

# Detections by class
print(f"\nDetections by Traffic Sign Class:")
print("-" * 70)
sign_counts = df['Traffic_Sign_Name'].value_counts().sort_values(ascending=False)
for sign, count in sign_counts.items():
    avg_conf = df[df['Traffic_Sign_Name'] == sign]['Confidence_Score'].mean()
    print(f"  {sign:.<40} {count:>3} detections (Avg: {avg_conf:.4f})")

# Detection status
print(f"\nDetection Status:")
print("-" * 70)
status_counts = df['Detection_Status'].value_counts()
for status, count in status_counts.items():
    print(f"  {status:.<40} {count:>3} detections")

# CSV file location
print(f"\nOutput Files:")
print("-" * 70)
if os.path.exists('detections.csv'):
    csv_size = os.path.getsize('detections.csv') / 1024
    print(f"  CSV File: detections.csv ({csv_size:.2f} KB)")

crops_dir = 'detections/crops'
if os.path.exists(crops_dir):
    crop_files = os.listdir(crops_dir)
    total_crop_size = sum(os.path.getsize(os.path.join(crops_dir, f)) for f in crop_files) / (1024*1024)
    print(f"  Cropped Images: {crops_dir}/ ({len(crop_files)} files, {total_crop_size:.2f} MB)")

print(f"\nCSV Structure ({len(df.columns)} columns):")
print("-" * 70)
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\nSample Detection (First Row):")
print("-" * 70)
first_det = df.iloc[0]
for col in ['Detection_ID', 'Timestamp', 'Traffic_Sign_Name', 'Confidence_Score', 
            'BoundingBox_X_Min', 'BoundingBox_Y_Min', 'BoundingBox_X_Max', 
            'BoundingBox_Y_Max', 'Crop_Image_Path']:
    print(f"  {col:.<40} {first_det[col]}")

print("\n" + "="*70)
print("PROJECT STATUS: ✓ SUCCESSFULLY COMPLETED".center(70))
print("="*70 + "\n")

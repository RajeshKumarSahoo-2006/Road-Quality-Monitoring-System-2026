# CSV Logging - Quick Reference

## Basic Commands

```bash
# Process single image (default)
python process_image.py

# Process all images in data/input
python process_image.py --all

# Process video (default location)
python process_video.py

# Process custom video
python process_video.py --video input.mp4
```

## Confidence Threshold Control

```bash
# Very strict (0.75 = few false positives)
python process_image.py --all --confidence 0.75

# Standard (0.65 = balanced)
python process_image.py --all --confidence 0.65

# Lenient (0.60 = catch more signs)
python process_image.py --all --confidence 0.60
```

## Video Processing

```bash
# No window, faster processing
python process_video.py --video input.mp4 --nogui

# Test first 100 frames
python process_video.py --video input.mp4 --max-frames 100

# Custom output paths
python process_video.py \
  --video input.mp4 \
  --output output.mp4 \
  --csv my_detections.csv
```

## CSV & Crops Management

```bash
# Custom CSV location
python process_image.py --all --csv logs/detections.csv

# Custom crops folder
python process_image.py --all --crops-dir results/crops

# Both custom
python process_video.py \
  --video input.mp4 \
  --csv output/detections.csv \
  --crops-dir output/sign_images
```

## Size Filtering

```bash
# Ignore boxes smaller than 30 pixels
python process_video.py --video input.mp4 --min-box-size 30

# Only large signs (50 pixels minimum)
python process_video.py --video input.mp4 --min-box-size 50
```

## Complete Examples

### Example 1: Standard Image Processing
```bash
python process_image.py --all --confidence 0.65
# Output: detections.csv + detections/crops/
```

### Example 2: Video with Custom Paths
```bash
python process_video.py \
  --video data/traffic.mp4 \
  --output runs/traffic_annotated.mp4 \
  --csv logs/traffic_detections.csv \
  --crops-dir logs/traffic_signs \
  --confidence 0.70 \
  --nogui
```

### Example 3: Quick Test
```bash
python process_video.py \
  --video input.mp4 \
  --max-frames 50 \
  --nogui
# Test with only 50 frames
```

### Example 4: High Precision (No False Positives)
```bash
python process_image.py \
  --all \
  --confidence 0.75 \
  --min-box-size 30
```

### Example 5: High Recall (Catch All Signs)
```bash
python process_image.py \
  --all \
  --confidence 0.60 \
  --min-box-size 20
```

## CSV File Columns

```
Detection_ID           - Unique ID (1, 2, 3...)
Timestamp              - YYYY-MM-DD HH:MM:SS
Frame_Number           - Video frame (0 for images)
Video_Source           - Filename or camera name
Traffic_Sign_Name      - "Speed Limit 50", "Stop", etc.
Confidence_Score       - 0.0 to 1.0
BoundingBox_X_Min      - Left X coordinate
BoundingBox_Y_Min      - Top Y coordinate
BoundingBox_X_Max      - Right X coordinate
BoundingBox_Y_Max      - Bottom Y coordinate
BoundingBox_Width      - Width in pixels
BoundingBox_Height     - Height in pixels
BoundingBox_Area       - Width × Height
Center_X               - Center X position
Center_Y               - Center Y position
Tracking_ID            - Object tracking ID
Detection_Status       - "Valid" or "Filtered"
Model_Name             - "YOLOv11"
Crop_Image_Path        - Path to cropped sign image
[... 10 more fields for future features ...]
```

## Analyzing Results

### Count detections by class
```python
import pandas as pd
df = pd.read_csv('detections.csv')
print(df['Traffic_Sign_Name'].value_counts())
```

### Average confidence per class
```python
print(df.groupby('Traffic_Sign_Name')['Confidence_Score'].mean())
```

### Valid detections only
```python
valid = df[df['Detection_Status'] == 'Valid']
print(f"Valid: {len(valid)}, Filtered: {len(df) - len(valid)}")
```

### High confidence detections
```python
high_conf = df[df['Confidence_Score'] >= 0.85]
print(f"High confidence (>0.85): {len(high_conf)}")
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CSV not created | Check folder write permissions |
| Crops not saved | Verify `detections/crops/` exists |
| Too many false positives | Increase `--confidence` to 0.75 |
| Missing real signs | Decrease `--confidence` to 0.60 |
| Slow processing | Use `--nogui` flag |
| Disk space full | Delete old crops: `rm -rf detections/crops` |

## File Structure After Processing

```
project/
├── detections.csv              ← Main results file
├── detections/
│   └── crops/                  ← Cropped sign images
│       ├── speed_limit_60_1542_1.jpg
│       ├── stop_sign_1543_2.jpg
│       └── ...
└── data/input/                 ← Your images/videos
```

## Parameter Summary

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `--confidence` | 0.65 | 0.60-0.75 | Detection threshold |
| `--csv` | detections.csv | path | CSV output file |
| `--crops-dir` | detections/crops | path | Cropped images folder |
| `--min-box-size` | 20 | pixels | Minimum detection size |
| `--nogui` | disabled | flag | Hide window |
| `--max-frames` | unlimited | number | Limit frames to test |

## Tips & Tricks

1. **Start High**: Begin with `--confidence 0.75`, lower if missing signs
2. **Check Crops**: Always inspect cropped images for quality
3. **Batch Export**: Use `--all` for multiple files
4. **Fast Test**: Use `--max-frames 50 --nogui` to test quickly
5. **Backup**: Save old CSV files before reprocessing
6. **Export**: Convert to Excel: `df.to_excel('report.xlsx')`
7. **Filter**: Load CSV and filter by confidence: `df[df['Confidence_Score'] > 0.7]`

---

For detailed information, see **CSV_LOGGING_GUIDE.md**

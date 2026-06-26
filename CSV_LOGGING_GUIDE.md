# CSV Logging System - Complete Guide

## Overview

This enhanced traffic sign detection system automatically logs all detections to a CSV file in real-time. Every detected traffic sign is recorded with comprehensive metadata for analysis and reporting.

---

## CSV Output Structure

### CSV File Location
- **Default**: `detections.csv` in project root
- **Cropped Images**: `detections/crops/` folder

### CSV Columns (28 Total)

| Column | Type | Description |
|--------|------|-------------|
| **Detection_ID** | Integer | Unique incremental ID for each detection |
| **Timestamp** | DateTime | YYYY-MM-DD HH:MM:SS format |
| **Frame_Number** | Integer | Frame number in video (0 for images) |
| **Video_Source** | String | Source file/camera/stream name |
| **Traffic_Sign_Name** | String | Class name (e.g., "Speed Limit 50") |
| **Traffic_Sign_Class_ID** | Integer | YOLO class ID |
| **Confidence_Score** | Float | Detection confidence (0.0-1.0) |
| **BoundingBox_X_Min** | Integer | Left edge X coordinate |
| **BoundingBox_Y_Min** | Integer | Top edge Y coordinate |
| **BoundingBox_X_Max** | Integer | Right edge X coordinate |
| **BoundingBox_Y_Max** | Integer | Bottom edge Y coordinate |
| **BoundingBox_Width** | Integer | Box width in pixels |
| **BoundingBox_Height** | Integer | Box height in pixels |
| **BoundingBox_Area** | Integer | Box area (width × height) |
| **Center_X** | Float | X coordinate of box center |
| **Center_Y** | Float | Y coordinate of box center |
| **Tracking_ID** | String | Object tracking ID (if enabled) |
| **Detection_Duration** | String | Time sign visible (future feature) |
| **Distance_Estimate** | String | Estimated distance (future feature) |
| **GPS_Latitude** | String | GPS latitude (future feature) |
| **GPS_Longitude** | String | GPS longitude (future feature) |
| **Vehicle_Speed** | String | Vehicle speed (future feature) |
| **Weather_Condition** | String | Weather info (customizable) |
| **Light_Condition** | String | Day/Night classification |
| **Detection_Status** | String | Valid/Filtered status |
| **Model_Name** | String | Detection model name |
| **Model_Version** | String | Model version |
| **Crop_Image_Path** | String | Path to cropped sign image |

---

## Example CSV Output

```csv
Detection_ID,Timestamp,Frame_Number,Video_Source,Traffic_Sign_Name,Traffic_Sign_Class_ID,Confidence_Score,BoundingBox_X_Min,BoundingBox_Y_Min,BoundingBox_X_Max,BoundingBox_Y_Max,BoundingBox_Width,BoundingBox_Height,BoundingBox_Area,Center_X,Center_Y,Tracking_ID,Detection_Duration,Distance_Estimate,GPS_Latitude,GPS_Longitude,Vehicle_Speed,Weather_Condition,Light_Condition,Detection_Status,Model_Name,Model_Version,Crop_Image_Path
1,2026-06-04 13:25:30,1542,video.mp4,Speed Limit 60,7,0.9423,320,145,412,238,92,93,8556,366.0,191.5,N/A,N/A,N/A,N/A,N/A,N/A,Clear,Day,Valid,YOLOv11,1.0,detections/crops/speed_limit_60_1542_1.jpg
2,2026-06-04 13:25:31,1543,video.mp4,Stop Sign,14,0.8942,198,267,301,362,103,95,9785,249.5,314.5,N/A,N/A,N/A,N/A,N/A,N/A,Clear,Day,Valid,YOLOv11,1.0,detections/crops/stop_sign_1543_2.jpg
3,2026-06-04 13:25:31,1543,video.mp4,Red Light,2,0.7654,450,112,502,178,52,66,3432,476.0,145.0,N/A,N/A,N/A,N/A,N/A,N/A,Clear,Day,Filtered,YOLOv11,1.0,N/A
```

---

## Filtering & Validation

### Confidence Threshold
- **Range**: 0.60 to 0.75 (safety limits)
- **Default**: 0.65
- **Effect**: Only detections above this score are logged as "Valid"
- **Below threshold**: Marked as "Filtered" but still logged

### Non-Maximum Suppression (NMS)
- **IoU Threshold**: 0.40-0.50 (prevents duplicate boxes)
- **Default**: 0.45
- **Effect**: Removes overlapping detections of same object

### Minimum Box Size
- **Default**: 20×20 pixels
- **Effect**: Ignores very small detections (likely false positives)
- **Result**: Marked as "Filtered"

---

## Usage Examples

### 1. Process Images with CSV Logging

#### Single Image
```bash
python process_image.py
```
- Uses default image: `data/input/stop_sign.jpg`
- Creates `detections.csv`
- Saves crops to `detections/crops/`

#### Batch Process All Images
```bash
python process_image.py --all
```
- Processes all images in `data/input/`
- One detection per image to CSV
- Shows/hides window with `--nogui`

#### Custom Confidence Threshold
```bash
python process_image.py --all --confidence 0.70
```
- More strict filtering (fewer false positives)
- Range: 0.60-0.75

#### Custom CSV Location
```bash
python process_image.py --all --csv custom/path/my_detections.csv
```

---

### 2. Process Videos with CSV Logging

#### Basic Video Detection
```bash
python process_video.py --video data/input/traffic_signs.mp4
```
- Logs all frame detections
- Saves annotated video
- Creates `detections.csv`

#### High-Confidence Detections Only
```bash
python process_video.py --video input.mp4 --confidence 0.75
```
- Filter weak detections
- Reduces false positives

#### Disable Window Display
```bash
python process_video.py --video input.mp4 --nogui
```
- Faster processing
- No OpenCV window
- Still saves results

#### Custom Output Paths
```bash
python process_video.py \
  --video input.mp4 \
  --output runs/detect/output.mp4 \
  --csv logs/detections.csv \
  --crops-dir logs/sign_crops
```

#### Quick Testing with Max Frames
```bash
python process_video.py --video input.mp4 --max-frames 100
```
- Process only first 100 frames
- Test pipeline quickly

---

### 3. Advanced Detection with detect_video.py

#### Standard Usage
```bash
python detect_video.py
```
- Auto-finds `Vdo.mp4` in project
- Saves to `output.mp4`
- Creates `detections.csv`

#### Full Configuration
```bash
python detect_video.py \
  --video data/input/traffic_video.mp4 \
  --output runs/detect/annotated.mp4 \
  --confidence 0.65 \
  --csv traffic_detections.csv \
  --crops-dir traffic_crops \
  --nogui
```

---

## Cropped Images

### Storage Location
- **Path**: `detections/crops/`
- **Naming**: `{class_name}_{frame_number}_{detection_id}.jpg`
- **Example**: `speed_limit_60_1542_1.jpg`

### Features
- Auto-cropped from detected regions
- Useful for manual verification
- Can be used for model retraining
- Referenced in CSV via `Crop_Image_Path`

### Managing Crops
```bash
# Delete old crops and start fresh
rm -rf detections/crops
python process_video.py --video new_video.mp4
```

---

## Understanding Detection Status

### Valid
- ✅ Confidence >= threshold
- ✅ Within frame boundaries
- ✅ Box size >= minimum
- ✅ Cropped image saved

### Filtered
- ⚠️ Confidence < threshold
- ⚠️ Box too small
- ⚠️ Outside boundaries
- ⚠️ Still logged but marked "Filtered"

### Invalid (Not Logged)
- ❌ NMS filtered out overlapping box
- ❌ Invalid coordinates

---

## Analyzing Results

### Python Analysis Example
```python
import pandas as pd

# Load CSV
df = pd.read_csv('detections.csv')

# Total detections
print(f"Total detections: {len(df)}")

# By class
print(df['Traffic_Sign_Name'].value_counts())

# By confidence
print(f"Average confidence: {df['Confidence_Score'].mean():.4f}")

# Valid only
valid = df[df['Detection_Status'] == 'Valid']
print(f"Valid detections: {len(valid)}")

# Export to Excel
df.to_excel('detections_report.xlsx', index=False)
```

### Query Examples
```python
# High confidence detections
high_conf = df[df['Confidence_Score'] >= 0.85]

# Speed limit signs only
speed_signs = df[df['Traffic_Sign_Name'].str.contains('Speed')]

# Frame-by-frame analysis
frame_stats = df.groupby('Frame_Number').size()
```

---

## Performance Tips

### Speed Up Processing
1. Use `--nogui` flag (no window display)
2. Increase confidence threshold (0.75)
3. Use `--max-frames` for testing
4. Disable video writer if not needed

### Reduce File Size
1. Use higher confidence threshold
2. Clean up old crop images
3. Archive old CSV files

### Better Accuracy
1. Lower confidence threshold (0.60)
2. Check cropped images manually
3. Verify CSV data

---

## Troubleshooting

### CSV Not Created
- Check write permissions in project folder
- Ensure `detections/` folder exists
- Try explicit path: `--csv ./results/detections.csv`

### Crops Not Saved
- Check `detections/crops/` folder permissions
- Verify disk space available
- Check image format support (JPEG)

### Wrong Frame Numbers
- Video streams sometimes skip frames
- CV2 frame count may differ from actual frames
- Use `Frame_Number` with timestamp for correlation

### High False Positives
- Increase confidence threshold: `--confidence 0.75`
- Reduce min-box-size: `--min-box-size 15`
- Check input video quality

---

## Advanced Features (Future)

The CSV structure supports future enhancements:

- **Tracking**: `Tracking_ID` column for multi-frame object tracking
- **GPS**: Latitude/Longitude integration
- **Telemetry**: Vehicle speed, direction
- **Duration**: How long sign visible in video
- **Distance**: Estimated distance to sign
- **Weather**: Automated weather condition logging

---

## Data Export & Integration

### Export to Other Formats
```python
import pandas as pd

df = pd.read_csv('detections.csv')

# Excel
df.to_excel('detections.xlsx', index=False)

# JSON
df.to_json('detections.json', orient='records')

# SQL Database
# from sqlalchemy import create_engine
# engine = create_engine('sqlite:///detections.db')
# df.to_sql('detections', engine, if_exists='append')
```

### Integration Examples
- Send alerts via email for Stop signs
- Update traffic management systems
- Feed data to autonomous vehicle systems
- Generate compliance reports
- Train better models on detected crops

---

## Support & Questions

For issues or questions:
1. Check CSV columns and data types
2. Verify confidence threshold is reasonable
3. Inspect cropped images for correctness
4. Use statistics summary (printed after processing)
5. Check file permissions and disk space

---

## Version Info

- **System**: Traffic Sign Detection with YOLOv11
- **CSV Logger**: Version 1.0
- **Model**: YOLOv11 (nano)
- **Last Updated**: June 2026

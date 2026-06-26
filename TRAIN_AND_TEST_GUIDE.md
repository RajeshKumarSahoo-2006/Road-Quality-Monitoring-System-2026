# Train and Test Pipeline - Quick Start Guide

## 🚀 Quick Start (3 Steps)

### Step 1: Prepare Your Dataset
Create the following folder structure:
```
dataset/
├── data.yaml (copy and fill the template)
├── train/
│   ├── images/ (your training images)
│   └── labels/ (corresponding .txt files)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Step 2: Run the Pipeline
```bash
python train_and_test.py
```

### Step 3: Check Results
```
results/
├── predictions/          # Annotated images
├── detection_results.csv # All detections
└── metrics_summary.json  # Training metrics
```

---

## 📝 Example YOLO Annotation Format

### Sample Image: traffic_light.jpg (640x480 pixels)

#### Detection: Red traffic light at coordinates (320, 240)
- Bounding box: x1=280, y1=200, x2=360, y2=280
- Normalized: x_center=0.5, y_center=0.5, width=0.125, height=0.167
- Class ID: 0 (Traffic Light Red)

**Label file (traffic_light.txt):**
```
0 0.5 0.5 0.125 0.167
```

#### Multiple Detections Example

Image: intersection.jpg (640x480 pixels) with 3 traffic signs:

1. **Red Traffic Light**
   - Position: top-left area
   - Normalized coords: 0.25, 0.2, 0.1, 0.15
   - Class: 0
   
2. **Speed Limit 50**
   - Position: right side
   - Normalized coords: 0.8, 0.4, 0.12, 0.12
   - Class: 3

3. **Stop Sign**
   - Position: bottom
   - Normalized coords: 0.5, 0.75, 0.15, 0.15
   - Class: 7

**Label file (intersection.txt):**
```
0 0.25 0.2 0.1 0.15
3 0.8 0.4 0.12 0.12
7 0.5 0.75 0.15 0.15
```

---

## 🎯 Class Reference

| ID | Class Name | Examples |
|----|---|---|
| 0 | Traffic Light Red | 🔴 Red traffic signals |
| 1 | Traffic Light Yellow | 🟡 Yellow traffic signals |
| 2 | Traffic Light Green | 🟢 Green traffic signals |
| 3 | Speed Limit Signs | Speed limit 25, 35, 50, etc. |
| 4 | Warning Signs | Yield, Pedestrian crossing, etc. |
| 5 | Direction Signs | Turn left, Turn right, etc. |
| 6 | Pedestrian Crossing | Pedestrian crossing signs |
| 7 | Stop Sign | 🛑 Stop signs |
| 8 | Yield Sign | Yield/Give way signs |
| 9 | No Entry Sign | No entry/Do not enter signs |

---

## ⚙️ Configuration Customization

### Modify Model Training Parameters

Edit the `main()` function in `train_and_test.py`:

```python
def main():
    # Configuration
    dataset_path = 'dataset'    # Your dataset location
    model_size = 'm'            # Model: 'n', 's', 'm', 'l', 'x'
    epochs = 100                # Training epochs
    batch_size = 16             # Batch size
```

### Model Size Selection

- **'n'**: Nano (fastest) - Good for real-time, low-accuracy requirements
- **'s'**: Small (fast) - Mobile/embedded systems
- **'m'**: Medium (default) - Balanced speed/accuracy
- **'l'**: Large (slower) - High accuracy, more resources
- **'x'**: XLarge (slowest) - Best accuracy, maximum resources

### Batch Size Recommendations

Based on GPU VRAM:
- 2GB VRAM: batch_size = 4
- 4GB VRAM: batch_size = 8
- 6GB VRAM: batch_size = 16
- 8GB+ VRAM: batch_size = 32

---

## 📊 Understanding CSV Output

### Sample detection_results.csv:

```csv
Image Name,Class Name,Confidence Score,xmin,ymin,xmax,ymax,Detection Time
traffic_light_001.jpg,Traffic Light Red,0.9542,125,245,189,312,0.0234
traffic_light_001.jpg,Speed Limit Signs,0.8732,250,180,290,220,0.0234
intersection_002.jpg,Stop Sign,0.9876,300,320,380,400,0.0189
speed_limit_003.jpg,Speed Limit Signs,0.9421,100,150,200,250,0.0201
```

### CSV Columns Explained:

- **Image Name**: Filename of the detected image
- **Class Name**: Type of traffic sign detected
- **Confidence Score**: Detection confidence (0-1, higher is better)
- **xmin, ymin**: Top-left corner of bounding box
- **xmax, ymax**: Bottom-right corner of bounding box
- **Detection Time**: Inference time in seconds

---

## 🔍 Validation Report Features

The pipeline automatically validates:

✅ **Dataset Structure**
- Verifies train/valid/test directories exist
- Checks for matching image-label pairs

✅ **Annotation Format**
- Validates YOLO coordinate format
- Ensures normalized values (0-1)
- Detects missing or malformed labels

✅ **Image Quality**
- Checks image dimensions
- Verifies readability
- Reports dimension statistics

✅ **Class Distribution**
- Shows instances per class
- Identifies class imbalance issues

---

## 📈 Output Metrics Explained

### Training Metrics:

- **Precision**: TP / (TP + FP) - Correctness of positive predictions
- **Recall**: TP / (TP + FN) - Coverage of actual positives
- **mAP@0.5**: Mean Average Precision at IoU=0.5
- **mAP@0.5:0.95**: Mean Average Precision across all IoU thresholds
- **F1-Score**: Harmonic mean of Precision and Recall

### Typical Performance Ranges:

```
Excellent:   Precision > 0.90, Recall > 0.90, mAP@0.5 > 0.85
Good:        Precision > 0.80, Recall > 0.80, mAP@0.5 > 0.75
Acceptable:  Precision > 0.70, Recall > 0.70, mAP@0.5 > 0.60
Needs Work:  Precision < 0.70 or Recall < 0.70
```

---

## 🐛 Common Issues & Solutions

### Issue: "No test images found!"
- **Cause**: test/images/ directory is empty
- **Solution**: Add test images to dataset/test/images/

### Issue: "data.yaml not found"
- **Cause**: data.yaml missing in dataset root
- **Solution**: Copy and fill the data.yaml template

### Issue: Missing labels warning
- **Cause**: Some images don't have corresponding .txt files
- **Solution**: 
  - Run validation to identify unpaired files
  - Either create missing labels or remove unpaired images

### Issue: Training takes too long
- **Cause**: Batch size too small or model too large
- **Solution**: 
  - Increase batch_size (if GPU memory allows)
  - Use smaller model size ('n' or 's')
  - Reduce epochs

### Issue: Out of Memory Error
- **Cause**: Batch size too large for GPU
- **Solution**: Reduce batch_size (try 8, 4, or 2)

### Issue: Poor accuracy on test set
- **Cause**: Insufficient/poor quality training data
- **Solution**:
  - Add more training images
  - Ensure accurate annotations
  - Balance class distribution
  - Increase epochs

---

## 🎓 Training Tips for Best Results

1. **Data Preparation**
   - Ensure consistent image quality
   - Balance class distribution
   - Verify annotation accuracy

2. **Training Strategy**
   - Start with 50 epochs, evaluate, then increase if needed
   - Use early stopping (patience parameter)
   - Monitor validation metrics

3. **Hyperparameter Tuning**
   - Adjust batch_size based on GPU memory
   - Fine-tune learning rate if needed
   - Experiment with augmentation settings

4. **Validation**
   - Use separate validation set (20% of data)
   - Monitor both train and validation loss
   - Watch for overfitting

---

## 📚 File Structure After Execution

```
project_root/
├── train_and_test.py           # Main script
├── data.yaml                   # Dataset config
├── dataset/                    # Your dataset
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
├── results/                    # Generated outputs
│   ├── predictions/            # Annotated images
│   ├── detection_results.csv
│   └── metrics_summary.json
├── runs/                       # YOLOv8 training outputs
│   └── detect/
│       └── train/
│           ├── weights/
│           ├── plots/
│           └── results.csv
├── training_pipeline.log       # Execution log
└── TRAINING_SETUP_GUIDE.md    # Full setup guide
```

---

## 🚀 Next Steps

1. **Prepare Dataset** → Organize images and labels
2. **Configure data.yaml** → Set correct paths
3. **Run Pipeline** → Execute `python train_and_test.py`
4. **Check Results** → Review outputs in results/ folder
5. **Deploy Model** → Use weights from runs/detect/train/weights/

---

## 📞 Getting Help

Check these files for more information:
- `TRAINING_SETUP_GUIDE.md` - Detailed setup instructions
- `training_pipeline.log` - Execution logs and error details
- `results/metrics_summary.json` - Complete metric details

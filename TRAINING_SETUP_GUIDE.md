# Traffic Sign Detection Training Pipeline - Setup Guide

## 📋 Overview

This guide explains how to prepare your dataset and run the complete YOLOv8 training pipeline.

## 🗂️ Required Dataset Structure

Before running the training script, organize your dataset as follows:

```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── labels/
│       ├── img_001.txt
│       ├── img_002.txt
│       └── ...
├── valid/
│   ├── images/
│   │   └── ...
│   └── labels/
│       └── ...
└── test/
    ├── images/
    │   └── ...
    └── labels/
        └── ...
```

## 📝 YOLO Annotation Format

Each `.txt` file should contain one annotation per line in the following format:

```
<class_id> <x_center> <y_center> <width> <height>
```

### Parameters:
- **class_id**: Integer (0-9) representing the traffic sign class
- **x_center, y_center, width, height**: Normalized coordinates (0-1) relative to image dimensions

### Example annotation:
```
0 0.5 0.5 0.3 0.4
2 0.2 0.3 0.15 0.2
5 0.8 0.7 0.2 0.25
```

## 🏷️ Traffic Sign Classes

The pipeline recognizes the following traffic sign classes:

| ID | Class Name |
|----|------------|
| 0 | Traffic Light Red |
| 1 | Traffic Light Yellow |
| 2 | Traffic Light Green |
| 3 | Speed Limit Signs |
| 4 | Warning Signs |
| 5 | Direction Signs |
| 6 | Pedestrian Crossing |
| 7 | Stop Sign |
| 8 | Yield Sign |
| 9 | No Entry Sign |

**Note**: You can modify these class names in the `TRAFFIC_SIGN_CLASSES` dictionary within the script.

## 📄 data.yaml Configuration

Create a `data.yaml` file in your dataset root directory:

```yaml
path: /absolute/path/to/dataset
train: train/images
val: valid/images
test: test/images

nc: 10  # Number of classes
names: ['Traffic Light Red', 'Traffic Light Yellow', 'Traffic Light Green', 
        'Speed Limit Signs', 'Warning Signs', 'Direction Signs',
        'Pedestrian Crossing', 'Stop Sign', 'Yield Sign', 'No Entry Sign']
```

## 🚀 Running the Training Pipeline

### Step 1: Prepare Your Dataset
- Organize images and labels according to the structure above
- Create `data.yaml` with your dataset paths

### Step 2: Run the Training Script
```bash
python train_and_test.py
```

### Step 3: Monitor Progress
The script will:
1. ✓ Install required packages automatically
2. ✓ Validate dataset structure and annotations
3. ✓ Train YOLOv8 model (with GPU if available)
4. ✓ Validate model performance
5. ✓ Run inference on test set
6. ✓ Generate annotated images
7. ✓ Export results to CSV

## ⚙️ Configuration Options

Modify these parameters in the `main()` function to customize the pipeline:

```python
dataset_path = 'dataset'    # Path to your dataset
model_size = 'm'            # YOLOv8 size: 'n', 's', 'm', 'l', 'x'
epochs = 100                # Number of training epochs
batch_size = 16             # Batch size (adjust based on GPU memory)
```

### Model Size Guide:
- **'n' (nano)**: Fastest, least accurate, ~3M parameters
- **'s' (small)**: Balance, ~11M parameters
- **'m' (medium)**: Default, ~25M parameters
- **'l' (large)**: Higher accuracy, ~53M parameters
- **'x' (xlarge)**: Best accuracy, ~71M parameters

## 📊 Output Files

After running the pipeline, you'll get:

### Directories:
- **results/predictions/**: Annotated images with bounding boxes
- **runs/detect/**: YOLOv8 training outputs (weights, plots, metrics)

### Files:
- **results/detection_results.csv**: All detections with coordinates and confidence
- **results/metrics_summary.json**: Training and validation metrics
- **training_pipeline.log**: Complete execution log

### CSV Format:
```
Image Name,Class Name,Confidence Score,xmin,ymin,xmax,ymax,Detection Time
image_001.jpg,Traffic Light Red,0.9542,125,245,189,312,0.0234
image_001.jpg,Speed Limit Signs,0.8732,250,180,290,220,0.0234
```

## 📈 Understanding the Output

### Metrics Explained:
- **Precision**: Percentage of correct predictions
- **Recall**: Percentage of detected traffic signs
- **mAP@0.5**: Mean Average Precision at 0.5 IOU threshold
- **mAP@0.5:0.95**: Mean Average Precision at all IOU thresholds
- **F1-Score**: Harmonic mean of precision and recall

### Confusion Matrix:
Shows true positives, false positives, and false negatives for each class.

## 🔧 Troubleshooting

### Dataset Not Found
```
Error: Dataset path not found: dataset
```
**Solution**: Create the `dataset` folder and organize files as shown above.

### Missing Labels
The validator will warn about images without corresponding label files. Add missing labels or remove unpaired images.

### Out of Memory (OOM) Error
Reduce `batch_size` in the main function:
```python
batch_size = 8  # Try smaller values: 4, 8, 16
```

### GPU Not Detected
The script automatically falls back to CPU. Ensure CUDA-compatible GPU drivers are installed.

### Validation Data Missing
If you don't have a validation set, split your training data:
```
train: 70% of your data
valid: 20% of your data
test: 10% of your data
```

## 💡 Tips for Better Results

1. **Data Quality**: Ensure annotations are accurate and complete
2. **Data Quantity**: More diverse images lead to better generalization
3. **Class Balance**: Ideally, each class should have similar representation
4. **Image Size**: Consistent image dimensions improve training
5. **Augmentation**: The script uses data augmentation automatically
6. **Epochs**: Start with 50-100 epochs and adjust based on validation metrics
7. **Batch Size**: Use the largest batch size your GPU memory allows

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow YOLO Format Guide](https://docs.roboflow.com/datasets/format-guides)
- [Traffic Sign Detection Datasets](https://www.roboflow.com/search?q=traffic%20sign)

## ✅ Validation Checklist

Before running the pipeline:
- [ ] Dataset folder exists
- [ ] All train/valid/test subdirectories created
- [ ] Images and labels paired correctly
- [ ] data.yaml file created with correct paths
- [ ] Class count (nc) matches your classes
- [ ] At least one image in each split

## 📞 Support

For issues with:
- **YOLOv8**: Check [Ultralytics Issues](https://github.com/ultralytics/ultralytics/issues)
- **Dataset Format**: Validate using the built-in `DatasetValidator` class
- **Training**: Check `training_pipeline.log` for detailed error messages

"""
Traffic Sign Detection and Recognition System using YOLOv8
Complete Production-Ready Training Pipeline

This script provides a comprehensive pipeline for:
1. Dataset structure and annotation validation
2. YOLOv8 model training with GPU support
3. Validation and testing on custom datasets
4. Metric generation and visualization
5. CSV export of detection results
6. Annotated image generation with bounding boxes

Author: Traffic Sign Detection Team
Date: 2024
"""

import os
import sys
import subprocess
import json
import csv
import warnings
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import shutil

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch

# Configure warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PackageManager:
    """Manages package installation and verification."""
    
    @staticmethod
    def install_required_packages():
        """Automatically install required packages if missing."""
        required_packages = {
            'torch': 'torch',
            'torchvision': 'torchvision',
            'ultralytics': 'ultralytics',
            'opencv-python': 'cv2',
            'numpy': 'numpy',
            'pillow': 'PIL',
            'matplotlib': 'matplotlib',
            'pandas': 'pandas'
        }
        
        logger.info("Checking required packages...")
        missing_packages = []
        
        for package_name, import_name in required_packages.items():
            try:
                __import__(import_name)
                logger.info(f"[OK] {package_name} is installed")
            except ImportError:
                logger.warning(f"[WARN] {package_name} is missing")
                missing_packages.append(package_name)
        
        if missing_packages:
            logger.info(f"Installing missing packages: {', '.join(missing_packages)}")
            for package in missing_packages:
                try:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", package],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    logger.info(f"✓ Successfully installed {package}")
                except subprocess.CalledProcessError:
                    logger.error(f"✗ Failed to install {package}")
                    sys.exit(1)
        
        logger.info("All required packages are installed!")
        return True


class DatasetValidator:
    """Validates dataset structure and YOLO annotation format."""
    
    # Standard traffic sign classes
    TRAFFIC_SIGN_CLASSES = {
        0: 'Traffic Light Red',
        1: 'Traffic Light Yellow',
        2: 'Traffic Light Green',
        3: 'Speed Limit Signs',
        4: 'Warning Signs',
        5: 'Direction Signs',
        6: 'Pedestrian Crossing',
        7: 'Stop Sign',
        8: 'Yield Sign',
        9: 'No Entry Sign'
    }
    
    def __init__(self, dataset_path: str):
        """
        Initialize dataset validator.
        
        Args:
            dataset_path: Path to dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.validation_report = {
            'total_images': 0,
            'total_labels': 0,
            'missing_labels': [],
            'missing_images': [],
            'invalid_annotations': [],
            'class_distribution': {},
            'image_stats': {'min_width': float('inf'), 'max_width': 0, 
                           'min_height': float('inf'), 'max_height': 0},
            'validation_passed': True
        }
    
    def validate_dataset_structure(self) -> bool:
        """
        Validate dataset directory structure.
        
        Returns:
            bool: True if structure is valid
        """
        logger.info(f"Validating dataset structure at {self.dataset_path}...")
        
        required_dirs = ['train/images', 'train/labels', 
                        'valid/images', 'valid/labels',
                        'test/images', 'test/labels']
        
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = self.dataset_path / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
                logger.warning(f"[WARN] Missing directory: {dir_path}")
        
        if missing_dirs:
            logger.error(f"Dataset structure validation failed. Missing: {missing_dirs}")
            self.validation_report['validation_passed'] = False
            return False
        
        logger.info("[OK] Dataset structure is valid")
        return True
    
    def validate_annotations(self) -> bool:
        """
        Validate YOLO annotation format.
        
        Returns:
            bool: True if all annotations are valid
        """
        logger.info("Validating YOLO annotations...")
        
        splits = ['train', 'valid', 'test']
        
        for split in splits:
            images_dir = self.dataset_path / split / 'images'
            labels_dir = self.dataset_path / split / 'labels'
            
            if not images_dir.exists():
                continue
            
            image_files = set([f.stem for f in images_dir.glob('*.*') 
                             if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            label_files = set([f.stem for f in labels_dir.glob('*.txt')])
            
            self.validation_report['total_images'] += len(image_files)
            self.validation_report['total_labels'] += len(label_files)
            
            # Check for missing labels
            missing_labels = image_files - label_files
            if missing_labels:
                self.validation_report['missing_labels'].extend(
                    [f"{split}/{name}" for name in missing_labels]
                )
                logger.warning(f"Missing labels for {len(missing_labels)} images in {split}")
            
            # Check for missing images
            missing_images = label_files - image_files
            if missing_images:
                self.validation_report['missing_images'].extend(
                    [f"{split}/{name}" for name in missing_images]
                )
                logger.warning(f"Missing images for {len(missing_images)} labels in {split}")
            
            # Validate label format
            for label_file in labels_dir.glob('*.txt'):
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        for line_num, line in enumerate(lines):
                            parts = line.strip().split()
                            if len(parts) != 5:
                                self.validation_report['invalid_annotations'].append(
                                    f"{split}/{label_file.name}:line {line_num + 1}"
                                )
                                logger.warning(f"Invalid format in {label_file.name}:{line_num + 1}")
                                continue
                            
                            try:
                                class_id = int(parts[0])
                                x_center, y_center, width, height = [float(p) for p in parts[1:]]
                                
                                # Validate normalized coordinates
                                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                       0 <= width <= 1 and 0 <= height <= 1):
                                    self.validation_report['invalid_annotations'].append(
                                        f"{split}/{label_file.name}:line {line_num + 1} (out of range)"
                                    )
                                    logger.warning(f"Coordinates out of range in {label_file.name}:{line_num + 1}")
                                
                                # Track class distribution
                                if class_id not in self.validation_report['class_distribution']:
                                    self.validation_report['class_distribution'][class_id] = 0
                                self.validation_report['class_distribution'][class_id] += 1
                                
                            except ValueError:
                                self.validation_report['invalid_annotations'].append(
                                    f"{split}/{label_file.name}:line {line_num + 1} (value error)"
                                )
                except Exception as e:
                    logger.error(f"Error reading {label_file}: {str(e)}")
            
            # Validate image properties
            for image_file in images_dir.glob('*.*'):
                try:
                    img = Image.open(image_file)
                    width, height = img.size
                    self.validation_report['image_stats']['min_width'] = \
                        min(self.validation_report['image_stats']['min_width'], width)
                    self.validation_report['image_stats']['max_width'] = \
                        max(self.validation_report['image_stats']['max_width'], width)
                    self.validation_report['image_stats']['min_height'] = \
                        min(self.validation_report['image_stats']['min_height'], height)
                    self.validation_report['image_stats']['max_height'] = \
                        max(self.validation_report['image_stats']['max_height'], height)
                except Exception as e:
                    logger.error(f"Error reading image {image_file}: {str(e)}")
        
        if self.validation_report['invalid_annotations']:
            self.validation_report['validation_passed'] = False
            logger.warning(f"Found {len(self.validation_report['invalid_annotations'])} invalid annotations")
        
        logger.info("[OK] Annotation validation complete")
        return self.validation_report['validation_passed']
    
    def print_validation_report(self):
        """Print comprehensive validation report."""
        logger.info("\n" + "="*80)
        logger.info("[DATASET VALIDATION REPORT]")
        logger.info("="*80)
        logger.info(f"Total Images: {self.validation_report['total_images']}")
        logger.info(f"Total Labels: {self.validation_report['total_labels']}")
        logger.info(f"Missing Labels: {len(self.validation_report['missing_labels'])}")
        logger.info(f"Missing Images: {len(self.validation_report['missing_images'])}")
        logger.info(f"Invalid Annotations: {len(self.validation_report['invalid_annotations'])}")
        
        logger.info("\nClass Distribution:")
        for class_id, count in sorted(self.validation_report['class_distribution'].items()):
            class_name = self.TRAFFIC_SIGN_CLASSES.get(class_id, f"Unknown (ID: {class_id})")
            logger.info(f"  {class_name}: {count} instances")
        
        logger.info("\nImage Statistics:")
        stats = self.validation_report['image_stats']
        logger.info(f"  Width range: {stats['min_width']} - {stats['max_width']} pixels")
        logger.info(f"  Height range: {stats['min_height']} - {stats['max_height']} pixels")
        
        logger.info(f"\nValidation Status: {'PASSED' if self.validation_report['validation_passed'] else 'FAILED'}")
        logger.info("="*80 + "\n")


class YOLOv8Trainer:
    """Manages YOLOv8 model training and evaluation."""
    
    def __init__(self, dataset_path: str, model_size: str = 'm', 
                 epochs: int = 100, batch_size: int = 16, patience: int = 20):
        """
        Initialize YOLOv8 trainer.
        
        Args:
            dataset_path: Path to dataset with data.yaml
            model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
            epochs: Number of training epochs
            batch_size: Batch size for training
            patience: Early stopping patience
        """
        self.dataset_path = Path(dataset_path)
        self.model_size = model_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = self._get_device()
        self.model = None
        self.results = {}
        
        try:
            from ultralytics import YOLO
            self.YOLO = YOLO
        except ImportError:
            logger.error("YOLOv8 not installed. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            from ultralytics import YOLO
            self.YOLO = YOLO
    
    def _get_device(self) -> str:
        """
        Detect available device (GPU or CPU).
        
        Returns:
            str: Device string ('cuda' or 'cpu')
        """
        if torch.cuda.is_available():
            device = 'cuda'
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"[OK] GPU detected: {device_name}")
        else:
            device = 'cpu'
            logger.info("GPU not available. Using CPU")
        
        return device
    
    def train_model(self) -> bool:
        """
        Train YOLOv8 model.
        
        Returns:
            bool: True if training completed successfully
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING YOLOV8 MODEL TRAINING")
        logger.info("="*80)
        logger.info(f"Model Size: {self.model_size}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"Device: {self.device}")
        logger.info("="*80 + "\n")
        
        try:
            # Load model
            model_name = f"yolov8{self.model_size}.pt"
            logger.info(f"Loading pre-trained model: {model_name}")
            self.model = self.YOLO(model_name)
            
            # Prepare data.yaml path
            data_yaml = self.dataset_path / 'data.yaml'
            if not data_yaml.exists():
                logger.error(f"data.yaml not found at {data_yaml}")
                return False
            
            logger.info(f"Using dataset config: {data_yaml}")
            
            # Train model
            logger.info("Starting training...")
            results = self.model.train(
                data=str(data_yaml),
                epochs=self.epochs,
                imgsz=640,
                batch=self.batch_size,
                patience=self.patience,
                device=self.device,
                augment=True,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=10.0,
                translate=0.1,
                scale=0.5,
                flipud=0.5,
                fliplr=0.5,
                mosaic=1.0,
                save=True,
                save_period=10,
                cache=True,
                val=True,
                plots=True
            )
            
            self.results = results
            logger.info("[OK] Training completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return False
    
    def validate_model(self) -> Dict:
        """
        Validate trained model.
        
        Returns:
            Dict: Validation metrics
        """
        if self.model is None:
            logger.error("Model not trained yet")
            return {}
        
        logger.info("\n" + "="*80)
        logger.info("VALIDATING MODEL")
        logger.info("="*80)
        
        try:
            # Validate on validation set
            data_yaml = self.dataset_path / 'data.yaml'
            metrics = self.model.val(data=str(data_yaml))
            
            validation_metrics = {
                'Precision': float(metrics.box.p.mean()) if hasattr(metrics.box, 'p') else 0,
                'Recall': float(metrics.box.r.mean()) if hasattr(metrics.box, 'r') else 0,
                'mAP@0.5': float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0,
                'mAP@0.5:0.95': float(metrics.box.map) if hasattr(metrics.box, 'map') else 0,
                'F1-Score': 0
            }
            
            # Calculate F1 Score
            if validation_metrics['Precision'] > 0 and validation_metrics['Recall'] > 0:
                p = validation_metrics['Precision']
                r = validation_metrics['Recall']
                validation_metrics['F1-Score'] = 2 * (p * r) / (p + r)
            
            logger.info("\nValidation Metrics:")
            for metric, value in validation_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            logger.info("="*80 + "\n")
            
            return validation_metrics
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return {}


class TestingPipeline:
    """Runs inference on test set and exports results."""
    
    # Traffic sign classes
    TRAFFIC_SIGN_CLASSES = {
        0: 'Traffic Light Red',
        1: 'Traffic Light Yellow',
        2: 'Traffic Light Green',
        3: 'Speed Limit Signs',
        4: 'Warning Signs',
        5: 'Direction Signs',
        6: 'Pedestrian Crossing',
        7: 'Stop Sign',
        8: 'Yield Sign',
        9: 'No Entry Sign'
    }
    
    def __init__(self, model, test_images_dir: str, output_dir: str = 'results'):
        """
        Initialize testing pipeline.
        
        Args:
            model: Trained YOLOv8 model
            test_images_dir: Directory containing test images
            output_dir: Directory for output files
        """
        self.model = model
        self.test_images_dir = Path(test_images_dir)
        self.output_dir = Path(output_dir)
        self.predictions_dir = self.output_dir / 'predictions'
        self.csv_path = self.output_dir / 'detection_results.csv'
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.predictions_dir.mkdir(exist_ok=True)
        
        self.detection_results = []
    
    def run_inference(self, conf: float = 0.5) -> bool:
        """
        Run inference on test images.
        
        Args:
            conf: Confidence threshold
            
        Returns:
            bool: True if inference completed successfully
        """
        logger.info("\n" + "="*80)
        logger.info("RUNNING INFERENCE ON TEST SET")
        logger.info("="*80)
        
        if not self.test_images_dir.exists():
            logger.error(f"Test images directory not found: {self.test_images_dir}")
            return False
        
        image_files = list(self.test_images_dir.glob('*.*'))
        image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        logger.info(f"Found {len(image_files)} test images")
        
        if len(image_files) == 0:
            logger.warning("No test images found!")
            return False
        
        for idx, image_path in enumerate(image_files, 1):
            try:
                start_time = datetime.now()
                
                # Run inference
                results = self.model.predict(
                    source=str(image_path),
                    conf=conf,
                    device=self.model.device,
                    verbose=False
                )
                
                detection_time = (datetime.now() - start_time).total_seconds()
                
                # Process results
                self._process_predictions(image_path, results[0], detection_time)
                
                # Progress logging
                if idx % max(1, len(image_files) // 10) == 0:
                    logger.info(f"Progress: {idx}/{len(image_files)} images processed")
                    
            except Exception as e:
                logger.error(f"Error processing {image_path.name}: {str(e)}")
                continue
        
        logger.info(f"[OK] Inference completed. Processed {len(self.detection_results)} images")
        logger.info("=" * 80 + "\n")
        return True
    
    def _process_predictions(self, image_path: Path, result, detection_time: float):
        """
        Process and store predictions for a single image.
        
        Args:
            image_path: Path to image file
            result: YOLO prediction result
            detection_time: Time taken for inference
        """
        # Load original image for annotation
        original_img = cv2.imread(str(image_path))
        if original_img is None:
            logger.error(f"Could not read image: {image_path}")
            return
        
        height, width = original_img.shape[:2]
        annotated_img = original_img.copy()
        
        # Process detections
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                try:
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    class_name = self.TRAFFIC_SIGN_CLASSES.get(
                        class_id, f"Unknown (ID: {class_id})"
                    )
                    
                    # Store detection result
                    self.detection_results.append({
                        'image_name': image_path.name,
                        'class_name': class_name,
                        'confidence': round(confidence, 4),
                        'xmin': int(x1),
                        'ymin': int(y1),
                        'xmax': int(x2),
                        'ymax': int(y2),
                        'detection_time': round(detection_time, 4)
                    })
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_img, (int(x1), int(y1)), 
                                (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{class_name} ({confidence:.2f})"
                    label_size, _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    cv2.rectangle(annotated_img, 
                                (int(x1), int(y1) - label_size[1] - 4),
                                (int(x1) + label_size[0], int(y1)), 
                                (0, 255, 0), -1)
                    
                    cv2.putText(annotated_img, label, 
                              (int(x1), int(y1) - 2),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                except Exception as e:
                    logger.error(f"Error processing detection: {str(e)}")
                    continue
        else:
            # No detections
            cv2.putText(annotated_img, "No Traffic Signs Detected", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Save annotated image
        output_path = self.predictions_dir / image_path.name
        cv2.imwrite(str(output_path), annotated_img)
    
    def export_to_csv(self) -> bool:
        """
        Export detection results to CSV file.
        
        Returns:
            bool: True if export was successful
        """
        logger.info("Exporting results to CSV...")
        
        try:
            if not self.detection_results:
                logger.warning("No detection results to export")
                return False
            
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['Image Name', 'Class Name', 'Confidence Score', 
                            'xmin', 'ymin', 'xmax', 'ymax', 'Detection Time']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                writer.writeheader()
                for result in self.detection_results:
                    writer.writerow({
                        'Image Name': result['image_name'],
                        'Class Name': result['class_name'],
                        'Confidence Score': result['confidence'],
                        'xmin': result['xmin'],
                        'ymin': result['ymin'],
                        'xmax': result['xmax'],
                        'ymax': result['ymax'],
                        'Detection Time': result['detection_time']
                    })
            
            logger.info(f"[OK] Results exported to {self.csv_path}")
            logger.info(f"  Total detections: {len(self.detection_results)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export CSV: {str(e)}")
            return False


class MetricsVisualizer:
    """Generates visualization of training and evaluation metrics."""
    
    def __init__(self, output_dir: str = 'results'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_metrics_summary(self, validation_metrics: Dict, 
                                detection_results: List[Dict]) -> bool:
        """
        Generate and save metrics summary.
        
        Args:
            validation_metrics: Validation metrics dictionary
            detection_results: List of detection results
            
        Returns:
            bool: True if summary was generated successfully
        """
        logger.info("Generating metrics summary...")
        
        try:
            summary_path = self.output_dir / 'metrics_summary.json'
            
            # Prepare summary data
            summary = {
                'timestamp': datetime.now().isoformat(),
                'validation_metrics': validation_metrics,
                'test_statistics': self._calculate_test_stats(detection_results),
                'detection_results_count': len(detection_results)
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=4)
            
            logger.info(f"[OK] Metrics summary saved to {summary_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate metrics summary: {str(e)}")
            return False
    
    def _calculate_test_stats(self, detection_results: List[Dict]) -> Dict:
        """
        Calculate test statistics.
        
        Args:
            detection_results: List of detection results
            
        Returns:
            Dict: Test statistics
        """
        if not detection_results:
            return {}
        
        confidences = [r['confidence'] for r in detection_results]
        detection_times = [r['detection_time'] for r in detection_results]
        
        class_counts = {}
        for result in detection_results:
            class_name = result['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'total_detections': len(detection_results),
            'unique_images': len(set(r['image_name'] for r in detection_results)),
            'average_confidence': round(np.mean(confidences), 4),
            'min_confidence': round(min(confidences), 4),
            'max_confidence': round(max(confidences), 4),
            'average_detection_time': round(np.mean(detection_times), 4),
            'total_detection_time': round(sum(detection_times), 4),
            'class_distribution': class_counts
        }


class TrainingPipeline:
    """Orchestrates the complete training and testing pipeline."""
    
    def __init__(self, dataset_path: str = 'dataset', model_size: str = 'm',
                 epochs: int = 100, batch_size: int = 16):
        """
        Initialize training pipeline.
        
        Args:
            dataset_path: Path to dataset
            model_size: YOLOv8 model size
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        self.dataset_path = dataset_path
        self.model_size = model_size
        self.epochs = epochs
        self.batch_size = batch_size
    
    def run_complete_pipeline(self) -> bool:
        """
        Run complete training and testing pipeline.
        
        Returns:
            bool: True if pipeline completed successfully
        """
        logger.info("\n" + "=" * 80)
        logger.info("=" + " " * 78 + "=")
        logger.info("=" + "TRAFFIC SIGN DETECTION - COMPLETE TRAINING PIPELINE".center(78) + "=")
        logger.info("=" + " " * 78 + "=")
        logger.info("=" * 80)
        
        # Step 1: Install packages
        if not PackageManager.install_required_packages():
            logger.error("Failed to install required packages")
            return False
        
        # Step 2: Validate dataset
        validator = DatasetValidator(self.dataset_path)
        if not validator.validate_dataset_structure():
            logger.error("Dataset structure validation failed")
            return False
        
        if not validator.validate_annotations():
            logger.warning("Dataset validation found issues, but continuing...")
        
        validator.print_validation_report()
        
        # Step 3: Train model
        trainer = YOLOv8Trainer(
            self.dataset_path,
            model_size=self.model_size,
            epochs=self.epochs,
            batch_size=self.batch_size
        )
        
        if not trainer.train_model():
            logger.error("Model training failed")
            return False
        
        # Step 4: Validate model
        validation_metrics = trainer.validate_model()
        
        # Step 5: Run inference on test set
        test_images_dir = Path(self.dataset_path) / 'test' / 'images'
        tester = TestingPipeline(trainer.model, str(test_images_dir), 'results')
        
        if not tester.run_inference(conf=0.5):
            logger.warning("Inference completed with issues")
        
        # Step 6: Export results
        tester.export_to_csv()
        
        # Step 7: Generate metrics
        visualizer = MetricsVisualizer('results')
        visualizer.generate_metrics_summary(validation_metrics, tester.detection_results)
        
        # Final summary
        self._print_final_summary(validation_metrics, tester.detection_results)
        
        logger.info("Training completed successfully!")
        return True
    
    def _print_final_summary(self, validation_metrics: Dict, detection_results: List[Dict]):
        """
        Print final pipeline summary.
        
        Args:
            validation_metrics: Model validation metrics
            detection_results: List of detection results
        """
        logger.info("\n" + "=" * 80)
        logger.info("[PIPELINE EXECUTION SUMMARY]")
        logger.info("=" * 80)
        
        logger.info("\n[VALIDATION METRICS]")
        for metric, value in validation_metrics.items():
            logger.info(f"  * {metric}: {value:.4f}")
        
        logger.info("\n[TEST RESULTS]")
        logger.info(f"  * Total detections: {len(detection_results)}")
        
        if detection_results:
            unique_images = len(set(r['image_name'] for r in detection_results))
            confidences = [r['confidence'] for r in detection_results]
            
            logger.info(f"  * Unique images with detections: {unique_images}")
            logger.info(f"  * Average confidence: {np.mean(confidences):.4f}")
            logger.info(f"  * Confidence range: {min(confidences):.4f} - {max(confidences):.4f}")
            
            # Class distribution
            class_counts = {}
            for result in detection_results:
                class_name = result['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            logger.info("\n  Class Distribution:")
            for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"    - {class_name}: {count}")
        
        logger.info("\n[OUTPUT FILES]")
        logger.info(f"  * Annotated images: results/predictions/")
        logger.info(f"  * CSV results: results/detection_results.csv")
        logger.info(f"  * Metrics summary: results/metrics_summary.json")
        logger.info(f"  * Training logs: training_pipeline.log")
        
        logger.info("\n" + "=" * 80)
        logger.info("[OK] PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80 + "\n")


def main():
    """Main entry point for training pipeline."""
    try:
        # Configuration
        dataset_path = 'dataset'  # Path to your dataset with data.yaml
        model_size = 'n'  # YOLOv8 model size: 'n', 's', 'm', 'l', 'x'
        epochs = 10  # Number of training epochs
        batch_size = 8  # Batch size
        
        # Verify dataset exists
        if not Path(dataset_path).exists():
            logger.error(f"Dataset path not found: {dataset_path}")
            logger.info("Please create a 'dataset' folder with the following structure:")
            logger.info("  dataset/")
            logger.info("  ├── data.yaml")
            logger.info("  ├── train/images/")
            logger.info("  ├── train/labels/")
            logger.info("  ├── valid/images/")
            logger.info("  ├── valid/labels/")
            logger.info("  ├── test/images/")
            logger.info("  └── test/labels/")
            return False
        
        # Run pipeline
        pipeline = TrainingPipeline(
            dataset_path=dataset_path,
            model_size=model_size,
            epochs=epochs,
            batch_size=batch_size
        )
        
        success = pipeline.run_complete_pipeline()
        return success
        
    except KeyboardInterrupt:
        logger.info("\n⚠ Pipeline interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

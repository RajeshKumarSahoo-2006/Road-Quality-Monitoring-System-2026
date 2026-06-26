"""
Simplified Traffic Sign Detection Training Script
This script provides a streamlined training pipeline with better error handling.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    logger.error("Required packages not found. Installing...")
    os.system(f"{sys.executable} -m pip install ultralytics torch torchvision")
    from ultralytics import YOLO
    import torch

def main():
    """Run simplified training pipeline"""
    
    logger.info("=" * 70)
    logger.info("TRAFFIC SIGN DETECTION - YOLOv8 TRAINING")
    logger.info("=" * 70)
    
    # Configuration
    dataset_path = Path('dataset')
    data_yaml = dataset_path / 'data.yaml'
    
    # Check dataset exists
    if not data_yaml.exists():
        logger.error(f"data.yaml not found at {data_yaml}")
        logger.info("Please ensure dataset structure exists with data.yaml")
        return False
    
    logger.info(f"\nDataset path: {dataset_path}")
    logger.info(f"Configuration: {data_yaml}")
    
    # Check GPU
    if torch.cuda.is_available():
        device = 0
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        logger.info("GPU not available, using CPU")
    
    try:
        logger.info("\n[STEP 1] Loading YOLOv8 model...")
        # Load model (this will download if not present)
        model = YOLO('yolov8n.pt')
        logger.info("Model loaded successfully")
        
        logger.info("\n[STEP 2] Starting training...")
        logger.info("Configuration:")
        logger.info("  - Model: YOLOv8 Nano")
        logger.info("  - Epochs: 10")
        logger.info("  - Batch Size: 8")
        logger.info("  - Device: " + ("GPU" if device != 'cpu' else "CPU"))
        
        # Train model
        results = model.train(
            data=str(data_yaml),
            epochs=10,
            imgsz=640,
            batch=8,
            device=device,
            patience=5,
            augment=True,
            save=True,
            plots=True,
            verbose=True
        )
        
        logger.info("\n[STEP 3] Training completed!")
        logger.info(f"Results saved to: {results}")
        
        logger.info("\n[STEP 4] Running validation...")
        # Validate
        metrics = model.val()
        logger.info(f"Validation metrics: {metrics}")
        
        logger.info("\n[STEP 5] Running inference on test set...")
        # Test
        test_dir = dataset_path / 'test' / 'images'
        if test_dir.exists() and any(test_dir.glob('*.*')):
            pred_results = model.predict(
                source=str(test_dir),
                conf=0.5,
                save=True,
                save_txt=True
            )
            logger.info(f"Predictions saved")
        else:
            logger.warning("No test images found")
        
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info("\nOutputs:")
        logger.info("  - Trained model: runs/detect/train/weights/best.pt")
        logger.info("  - Predictions: runs/detect/predict/")
        logger.info("  - Training plots: runs/detect/train/")
        
        return True
        
    except Exception as e:
        logger.error(f"\nTraining failed: {str(e)}")
        logger.info("\nTroubleshooting:")
        logger.info("  1. Check if data.yaml path is correct")
        logger.info("  2. Verify dataset structure (train/images, train/labels, etc.)")
        logger.info("  3. Ensure internet connection for model download")
        logger.info("  4. Check GPU drivers if using CUDA")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

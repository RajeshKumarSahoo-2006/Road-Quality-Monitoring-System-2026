import csv
import os
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, Optional, Tuple


class DetectionCSVLogger:
    """
    Comprehensive CSV logger for traffic sign detections.
    Handles real-time logging, validation, and image cropping.
    """
    
    # CSV columns in order
    CSV_COLUMNS = [
        'Detection_ID',
        'Timestamp',
        'Frame_Number',
        'Video_Source',
        'Traffic_Sign_Name',
        'Traffic_Sign_Class_ID',
        'Confidence_Score',
        'BoundingBox_X_Min',
        'BoundingBox_Y_Min',
        'BoundingBox_X_Max',
        'BoundingBox_Y_Max',
        'BoundingBox_Width',
        'BoundingBox_Height',
        'BoundingBox_Area',
        'Center_X',
        'Center_Y',
        'Tracking_ID',
        'Detection_Duration',
        'Distance_Estimate',
        'GPS_Latitude',
        'GPS_Longitude',
        'Vehicle_Speed',
        'Weather_Condition',
        'Light_Condition',
        'Detection_Status',
        'Model_Name',
        'Model_Version',
        'Crop_Image_Path'
    ]
    
    # Default configuration
    CONFIDENCE_THRESHOLD_MIN = 0.60
    CONFIDENCE_THRESHOLD_MAX = 0.75
    MIN_BOX_SIZE = 20  # pixels
    NMS_IOU_THRESHOLD = 0.45
    
    def __init__(
        self, 
        csv_file: str = "detections.csv",
        crops_dir: str = "detections/crops",
        confidence_threshold: float = 0.65,
        min_box_size: int = 20,
        nms_iou: float = 0.45,
        model_name: str = "YOLOv11",
        model_version: str = "1.0"
    ):
        """
        Initialize the CSV logger.
        
        Args:
            csv_file: Path to CSV file
            crops_dir: Directory to save cropped images
            confidence_threshold: Minimum confidence to log (0.60-0.75)
            min_box_size: Minimum bounding box size in pixels
            nms_iou: NMS IoU threshold
            model_name: Name of detection model
            model_version: Version of detection model
        """
        self.csv_file = csv_file
        self.crops_dir = crops_dir
        self.confidence_threshold = max(
            self.CONFIDENCE_THRESHOLD_MIN,
            min(confidence_threshold, self.CONFIDENCE_THRESHOLD_MAX)
        )
        self.min_box_size = min_box_size
        self.nms_iou = nms_iou
        self.model_name = model_name
        self.model_version = model_version
        self.detection_id_counter = 0
        self.tracked_detections = {}  # Track active detections
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.crops_dir) or ".", exist_ok=True)
        os.makedirs(self.crops_dir, exist_ok=True)
        
        # Initialize CSV file
        self._initialize_csv()
        self._load_detection_counter()
    
    def _initialize_csv(self):
        """Create CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_file):
            try:
                with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
                    writer.writeheader()
                print(f"Created CSV file: {self.csv_file}")
            except Exception as e:
                print(f"Error creating CSV file: {e}")
    
    def _load_detection_counter(self):
        """Load the next detection ID from existing CSV."""
        if os.path.exists(self.csv_file) and os.path.getsize(self.csv_file) > 0:
            try:
                with open(self.csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        last_id = int(rows[-1]['Detection_ID'])
                        self.detection_id_counter = last_id
            except Exception as e:
                print(f"Error loading detection counter: {e}")
    
    def _apply_nms(self, detections: list) -> list:
        """
        Apply Non-Maximum Suppression to remove overlapping boxes.
        
        Args:
            detections: List of detection dicts with boxes
            
        Returns:
            Filtered detections after NMS
        """
        if not detections:
            return []
        
        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda x: float(x['conf']), reverse=True)
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # Remove boxes with high IoU with current box
            remaining = []
            for det in detections:
                iou = self._calculate_iou(current, det)
                if iou < self.nms_iou:
                    remaining.append(det)
            detections = remaining
        
        return keep
    
    def _calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """Calculate Intersection over Union (IoU) between two boxes."""
        try:
            x1_min, y1_min, x1_max, y1_max = (
                float(box1['x1']), float(box1['y1']),
                float(box1['x2']), float(box1['y2'])
            )
            x2_min, y2_min, x2_max, y2_max = (
                float(box2['x1']), float(box2['y1']),
                float(box2['x2']), float(box2['y2'])
            )
            
            # Calculate intersection
            inter_xmin = max(x1_min, x2_min)
            inter_ymin = max(y1_min, y2_min)
            inter_xmax = min(x1_max, x2_max)
            inter_ymax = min(y1_max, y2_max)
            
            if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
                return 0.0
            
            inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
            
            # Calculate union
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = box1_area + box2_area - inter_area
            
            if union_area == 0:
                return 0.0
            
            return inter_area / union_area
        except Exception:
            return 0.0
    
    def _validate_detection(
        self,
        x1: float, y1: float, x2: float, y2: float,
        confidence: float,
        width: int, height: int
    ) -> Tuple[bool, str]:
        """
        Validate detection before logging.
        
        Returns:
            (is_valid, status_message)
        """
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return False, f"Confidence {confidence:.2f} below threshold {self.confidence_threshold}"
        
        # Check bounding box coordinates
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return False, "Bounding box outside frame boundaries"
        
        if x1 >= x2 or y1 >= y2:
            return False, "Invalid bounding box coordinates"
        
        # Check minimum box size
        box_width = int(x2 - x1)
        box_height = int(y2 - y1)
        
        if box_width < self.min_box_size or box_height < self.min_box_size:
            return False, f"Box too small ({box_width}x{box_height}, min {self.min_box_size})"
        
        return True, "Valid"
    
    def _crop_detection(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        class_name: str,
        frame_number: int
    ) -> Optional[str]:
        """
        Crop and save the detected region.
        
        Returns:
            Path to cropped image or None if failed
        """
        try:
            cropped = frame[int(y1):int(y2), int(x1):int(x2)]
            
            if cropped.size == 0:
                return None
            
            # Create filename
            safe_class_name = class_name.replace(" ", "_").lower()
            filename = f"{safe_class_name}_{frame_number}_{self.detection_id_counter}.jpg"
            filepath = os.path.join(self.crops_dir, filename)
            
            # Save crop
            cv2.imwrite(filepath, cropped)
            return filepath
        except Exception as e:
            print(f"Error cropping detection: {e}")
            return None
    
    def log_detections(
        self,
        frame: np.ndarray,
        detections: list,
        frame_number: int = 0,
        video_source: str = "Unknown",
        class_names: Dict = None,
        light_condition: str = "Unknown",
        weather_condition: str = "Unknown"
    ) -> int:
        """
        Process and log multiple detections from a frame.
        
        Args:
            frame: Input frame/image
            detections: List of detection dicts with keys:
                       {x1, y1, x2, y2, conf, class_id, class_name}
            frame_number: Frame number for video
            video_source: Source identifier (filename, camera, etc.)
            class_names: Dict mapping class_id to class_name
            light_condition: "Day" or "Night"
            weather_condition: Weather description
            
        Returns:
            Number of detections logged
        """
        if not detections:
            return 0
        
        height, width = frame.shape[:2]
        logged_count = 0
        
        # Apply NMS filtering
        filtered_detections = self._apply_nms(detections)
        
        for det in filtered_detections:
            try:
                x1 = float(det['x1'])
                y1 = float(det['y1'])
                x2 = float(det['x2'])
                y2 = float(det['y2'])
                confidence = float(det['conf'])
                class_id = int(det.get('class_id', -1))
                class_name = det.get('class_name', 'Unknown')
                tracking_id = det.get('tracking_id', None)
                
                # Validate detection
                is_valid, status = self._validate_detection(
                    x1, y1, x2, y2, confidence, width, height
                )
                
                if not is_valid:
                    continue
                
                # Calculate metrics
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Crop image
                crop_path = self._crop_detection(
                    frame, int(x1), int(y1), int(x2), int(y2),
                    class_name, frame_number
                )
                
                # Prepare row data
                self.detection_id_counter += 1
                row = {
                    'Detection_ID': self.detection_id_counter,
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Frame_Number': frame_number,
                    'Video_Source': video_source,
                    'Traffic_Sign_Name': class_name,
                    'Traffic_Sign_Class_ID': class_id,
                    'Confidence_Score': round(confidence, 4),
                    'BoundingBox_X_Min': int(x1),
                    'BoundingBox_Y_Min': int(y1),
                    'BoundingBox_X_Max': int(x2),
                    'BoundingBox_Y_Max': int(y2),
                    'BoundingBox_Width': int(box_width),
                    'BoundingBox_Height': int(box_height),
                    'BoundingBox_Area': int(box_area),
                    'Center_X': round(center_x, 2),
                    'Center_Y': round(center_y, 2),
                    'Tracking_ID': tracking_id or 'N/A',
                    'Detection_Duration': 'N/A',
                    'Distance_Estimate': 'N/A',
                    'GPS_Latitude': 'N/A',
                    'GPS_Longitude': 'N/A',
                    'Vehicle_Speed': 'N/A',
                    'Weather_Condition': weather_condition,
                    'Light_Condition': light_condition,
                    'Detection_Status': status,
                    'Model_Name': self.model_name,
                    'Model_Version': self.model_version,
                    'Crop_Image_Path': crop_path or 'N/A'
                }
                
                # Write to CSV
                self._write_row(row)
                logged_count += 1
                
            except Exception as e:
                print(f"Error logging detection: {e}")
                continue
        
        return logged_count
    
    def _write_row(self, row: Dict):
        """Write a single detection row to CSV."""
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
                writer.writerow(row)
        except Exception as e:
            print(f"Error writing to CSV: {e}")
    
    def get_statistics(self) -> Dict:
        """Get statistics from current CSV file."""
        stats = {
            'total_detections': 0,
            'detections_by_class': {},
            'avg_confidence': 0,
            'unique_frames': set()
        }
        
        if not os.path.exists(self.csv_file):
            return stats
        
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                confidences = []
                
                for row in reader:
                    stats['total_detections'] += 1
                    class_name = row.get('Traffic_Sign_Name', 'Unknown')
                    stats['detections_by_class'][class_name] = \
                        stats['detections_by_class'].get(class_name, 0) + 1
                    
                    try:
                        stats['unique_frames'].add(int(row.get('Frame_Number', 0)))
                        confidences.append(float(row.get('Confidence_Score', 0)))
                    except ValueError:
                        pass
                
                if confidences:
                    stats['avg_confidence'] = round(sum(confidences) / len(confidences), 4)
                
                stats['unique_frames'] = len(stats['unique_frames'])
                
        except Exception as e:
            print(f"Error reading statistics: {e}")
        
        return stats
    
    def print_summary(self):
        """Print detection summary to console."""
        stats = self.get_statistics()
        print("\n" + "="*60)
        print("DETECTION SUMMARY")
        print("="*60)
        print(f"Total Detections: {stats['total_detections']}")
        print(f"Unique Frames: {stats['unique_frames']}")
        print(f"Average Confidence: {stats['avg_confidence']:.4f}")
        print(f"\nDetections by Class:")
        for class_name, count in sorted(
            stats['detections_by_class'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"  {class_name}: {count}")
        print(f"\nCSV File: {self.csv_file}")
        print(f"Crops Directory: {self.crops_dir}")
        print("="*60 + "\n")

import argparse
import cv2
import os
from ultralytics import YOLO
from csv_logger import DetectionCSVLogger


def parse_args():
    parser = argparse.ArgumentParser(description='Traffic sign detection on images')
    parser.add_argument('--nogui', action='store_true', help='Disable image window display')
    parser.add_argument('--all', action='store_true', help='Process all images in data/input directory')
    parser.add_argument('--confidence', type=float, default=0.65, 
                       help='Confidence threshold (0.60-0.75, default 0.65)')
    parser.add_argument('--csv', default='detections.csv', help='CSV file path')
    parser.add_argument('--crops-dir', default='detections/crops', help='Crops directory path')
    parser.add_argument('--min-box-size', type=int, default=20, help='Minimum bounding box size')
    return parser.parse_args()


def main():
    args = parse_args()
    detector = YOLO("./model/traffic_sign_detector.pt", task="detect")
    
    # Initialize CSV logger
    csv_logger = DetectionCSVLogger(
        csv_file=args.csv,
        crops_dir=args.crops_dir,
        confidence_threshold=args.confidence,
        min_box_size=args.min_box_size
    )
    
    if args.all:
        # Process all images in data/input directory
        input_dir = "./data/input"
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return
        
        print(f"Found {len(image_files)} images to process")
        for img_file in image_files:
            img_path = os.path.join(input_dir, img_file)
            process_single_image(detector, img_path, args, csv_logger)
    else:
        # Process single default image
        img_path = "./data/input/stop_sign.jpg"
        process_single_image(detector, img_path, args, csv_logger)
    
    # Print summary
    csv_logger.print_summary()


def process_single_image(detector, img_path, args, csv_logger):
    image = cv2.imread(img_path)
    if image is None:
        print(f"Unable to read image from {img_path}")
        return

    detections = detector(image)
    total_boxes = 0
    logged_boxes = 0
    
    # Prepare detections for CSV logger
    detection_list = []
    
    for detection in detections:
        class_ids = detection.boxes.cls
        for i, bbox in enumerate(detection.boxes):
            x1, y1, x2, y2 = bbox.xyxy[0]
            confidence = float(bbox.conf[0])
            class_id = int(class_ids[i])
            class_name = detection.names[class_id]
            
            detection_list.append({
                'x1': float(x1),
                'y1': float(y1),
                'x2': float(x2),
                'y2': float(y2),
                'conf': confidence,
                'class_id': class_id,
                'class_name': class_name
            })
            total_boxes += 1
    
    # Log detections to CSV
    if detection_list:
        filename = os.path.basename(img_path)
        logged_boxes = csv_logger.log_detections(
            frame=image,
            detections=detection_list,
            frame_number=0,
            video_source=filename,
            light_condition="Day"
        )
    
    # Draw boxes on image (after filtering through CSV logger)
    for det in detection_list:
        x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
        conf = det['conf']
        class_name = det['class_name']
        
        # Only draw if passed validation
        if conf >= args.confidence:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if args.nogui:
        print(f"Processed {img_path}. Detected {total_boxes} boxes, Logged {logged_boxes} boxes.")
    else:
        cv2.imshow(f"Detected - {img_path}", image)
        print(f"Showing {img_path} with {logged_boxes}/{total_boxes} valid boxes. Press any key to continue.")
        cv2.waitKey(0)


if __name__ == "__main__":
    main()

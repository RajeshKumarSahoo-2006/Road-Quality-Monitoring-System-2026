import argparse
import collections
import os
import sys
import traceback
import cv2
from ultralytics import YOLO
from csv_logger import DetectionCSVLogger


def parse_args():
    parser = argparse.ArgumentParser(description='Traffic sign detection on a video')
    parser.add_argument('--nogui', action='store_true', help='Disable video window display')
    parser.add_argument('--input', '-i', default=None, help='Path to input video (overrides default)')
    parser.add_argument('--output', '-o', default=None, help='Path to output annotated video')
    parser.add_argument('--confidence', type=float, default=0.65,
                       help='Confidence threshold (0.60-0.75, default 0.65)')
    parser.add_argument('--csv', default='detections.csv', help='CSV file path')
    parser.add_argument('--crops-dir', default='detections/crops', help='Crops directory path')
    parser.add_argument('--min-box-size', type=int, default=20, help='Minimum bounding box size')
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        detector = YOLO("./model/traffic_sign_detector.pt", task="detect")
    except Exception:
        print("Failed to load model:\n" + traceback.format_exc())
        return
    
    # Initialize CSV logger
    csv_logger = DetectionCSVLogger(
        csv_file=args.csv,
        crops_dir=args.crops_dir,
        confidence_threshold=args.confidence,
        min_box_size=args.min_box_size
    )
    
    class_names = getattr(detector, 'names', {}) or {}
    video_path = args.input or "D:\\DeepSurge\\traffic-sign-detection-using-yolov11-main\\traffic-sign-detection-using-yolov11-main\\data\\input\\traffic_signs.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Unable to open the input file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    output_path = args.output or os.path.join("./runs/detect", "predict_video.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = None
    if width > 0 and height > 0:
        try:
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                writer = None
        except Exception:
            writer = None

    frame_count = 0
    detections_total = 0
    detections_logged = 0
    detections_by_class = collections.Counter()
    
    video_filename = os.path.basename(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            results = detector(frame)
        except Exception:
            print("Warning: detector failed on a frame:\n" + traceback.format_exc())
            results = []
        
        detection_list = []
        
        if len(results) > 0:
            result = results[0]
            for bbox in result.boxes:
                try:
                    xy = bbox.xyxy[0]
                    x1, y1, x2, y2 = float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])
                except Exception:
                    continue
                # safe scalar extraction
                def _to_scalar(v):
                    try:
                        return float(v.item())
                    except Exception:
                        try:
                            return float(v[0])
                        except Exception:
                            return float(v)

                try:
                    cls_raw = getattr(bbox, 'cls', None)
                    cls_id = int(_to_scalar(cls_raw)) if cls_raw is not None else -1
                except Exception:
                    cls_id = -1
                try:
                    conf_raw = getattr(bbox, 'conf', None)
                    conf = float(_to_scalar(conf_raw)) if conf_raw is not None else 0.0
                except Exception:
                    conf = 0.0
                
                class_name = class_names.get(cls_id, str(cls_id))
                
                detection_list.append({
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'conf': conf,
                    'class_id': cls_id,
                    'class_name': class_name
                })
                detections_total += 1
        
        # Log detections to CSV
        if detection_list:
            logged = csv_logger.log_detections(
                frame=frame,
                detections=detection_list,
                frame_number=frame_count,
                video_source=video_filename,
                light_condition="Day"
            )
            detections_logged += logged
        
        # Draw boxes on frame for visualization
        for det in detection_list:
            x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
            conf = det['conf']
            class_name = det['class_name']
            
            # Only draw if passed validation
            if conf >= args.confidence:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                detections_by_class[class_name] += 1

        frame_count += 1
        if writer is not None:
            writer.write(frame)
        if not args.nogui:
            cv2.imshow("Traffic sign detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if not args.nogui:
        cv2.destroyAllWindows()
    
    print(f"\nProcessed {frame_count} frames, detected {detections_total} boxes, logged {detections_logged} boxes.")
    print(f"Annotated video saved to: {output_path}")
    if detections_by_class:
        print("Detection summary:")
        for class_name, count in detections_by_class.items():
            print(f"  {class_name}: {count}")
    
    # Print CSV summary
    csv_logger.print_summary()


if __name__ == "__main__":
    main()
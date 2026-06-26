import argparse
import os
import sys

import cv2
from ultralytics import YOLO
from csv_logger import DetectionCSVLogger


def find_video_path(video_name="Vdo.mp4"):
    """Search for the video file in common locations inside the project."""
    possible_paths = [
        video_name,
        os.path.join("data", "input", video_name),
        os.path.join("data", "input", video_name.lower()),
        os.path.join("data", "input", video_name.upper()),
        os.path.join(".", video_name.lower()),
        os.path.join(".", video_name.upper()),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Could not find the video. Tried: {', '.join(possible_paths)}"
    )


def build_output_path(output_name="output.mp4"):
    output_path = os.path.abspath(output_name)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    return output_path


def process_video(
    video_path,
    model_path="./model/traffic_sign_detector.pt",
    output_path="output.mp4",
    show_window=True,
    max_frames=None,
    csv_file="detections.csv",
    crops_dir="detections/crops",
    confidence_threshold=0.65,
    min_box_size=20,
):
    """Run YOLO detection on a video and save the annotated result."""
    model = YOLO(model_path)
    
    # Initialize CSV logger
    csv_logger = DetectionCSVLogger(
        csv_file=csv_file,
        crops_dir=crops_dir,
        confidence_threshold=confidence_threshold,
        min_box_size=min_box_size
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Unable to initialize video writer for: {output_path}")

    frame_count = 0
    total_boxes = 0
    logged_boxes = 0

    print(f"Processing video: {video_path}")
    print(f"Saving output to: {output_path}")
    
    video_filename = os.path.basename(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        result = results[0]
        
        detection_list = []

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result.names[cls_id]
            
            detection_list.append({
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'conf': conf,
                'class_id': cls_id,
                'class_name': class_name
            })
            total_boxes += 1
        
        # Log detections to CSV
        if detection_list:
            logged = csv_logger.log_detections(
                frame=frame,
                detections=detection_list,
                frame_number=frame_count,
                video_source=video_filename,
                light_condition="Day"
            )
            logged_boxes += logged
        
        # Draw boxes on frame (only valid ones)
        for det in detection_list:
            x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
            conf = det['conf']
            class_name = det['class_name']
            
            if conf >= confidence_threshold:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        writer.write(frame)

        if show_window:
            cv2.imshow("Traffic Sign Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Exit key pressed. Stopping early.")
                break

        frame_count += 1
        if max_frames is not None and frame_count >= max_frames:
            break

    cap.release()
    writer.release()

    if show_window:
        cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames, detected {total_boxes} boxes, logged {logged_boxes} boxes.")
    print(f"Total detections: {logged_boxes}")
    print(f"Output saved to: {output_path}")
    
    # Print CSV summary
    csv_logger.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="Detect traffic signs, symbols, and road sign boards in a video using YOLO"
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Path to the input video. Defaults to searching for Vdo.mp4 in the project.",
    )
    parser.add_argument(
        "--model",
        default="./model/traffic_sign_detector.pt",
        help="Path to the YOLO model file.",
    )
    parser.add_argument(
        "--output",
        default="output.mp4",
        help="Output video file name. Default: output.mp4",
    )
    parser.add_argument(
        "--nogui",
        action="store_true",
        help="Disable the OpenCV window and process the video without showing it.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional: limit the number of frames processed for quick testing.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.65,
        help="Confidence threshold (0.60-0.75, default 0.65)",
    )
    parser.add_argument(
        "--csv",
        default="detections.csv",
        help="CSV file path for logging detections",
    )
    parser.add_argument(
        "--crops-dir",
        default="detections/crops",
        help="Directory for saving cropped detection images",
    )
    parser.add_argument(
        "--min-box-size",
        type=int,
        default=20,
        help="Minimum bounding box size in pixels",
    )

    args = parser.parse_args()

    try:
        video_path = args.video or find_video_path("Vdo.mp4")
        output_path = build_output_path(args.output)
        process_video(
            video_path=video_path,
            model_path=args.model,
            output_path=output_path,
            show_window=not args.nogui,
            max_frames=args.max_frames,
            csv_file=args.csv,
            crops_dir=args.crops_dir,
            confidence_threshold=args.confidence,
            min_box_size=args.min_box_size,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

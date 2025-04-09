import cv2
from ultralytics import YOLO

def run_inference_on_video(model_path, input_video_path, output_video_path, conf_threshold=0.5):
    # Load the YOLO model
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize video writer
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on the frame
        results = model(frame, conf=conf_threshold)

        # Draw bounding boxes on the frame
        annotated_frame = results[0].plot()

        # Write the frame with bounding boxes to the output video
        out.write(annotated_frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Inference complete. Output video saved to {output_video_path}")
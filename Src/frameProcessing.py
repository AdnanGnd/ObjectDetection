import cv2
import numpy as np
from torchvision.transforms import ToTensor
import torch

def process_frame(frame, model, device, CLASSES):
    # Preprocess the frame
    input_size = (224, 224)  # Match the input size expected by the model
    frame_resized = cv2.resize(frame, input_size)
    frame_tensor = ToTensor()(frame_resized).unsqueeze(0).to(device)  # Add batch dimension

    # Run the model
    with torch.no_grad():
        class_output, box_output = model(frame_tensor)

    # Convert predictions to class labels and bounding boxes
    class_ids = torch.argmax(class_output, dim=1).cpu().numpy()  # Predicted classes
    bboxes = box_output.cpu().numpy()  # Bounding box coordinates [x_min, y_min, x_max, y_max]

    # Scale bounding box coordinates back to the original frame size
    height, width, _ = frame.shape
    bboxes = bboxes * np.array([width, height, width, height])  # Scale to original frame size
    bboxes = bboxes.astype(int)  # Convert to integers

    # Draw bounding boxes and class labels on the frame
    for class_id, bbox in zip(class_ids, bboxes):
        x_min, y_min, x_max, y_max = bbox
        class_label = CLASSES[class_id]  # Class name
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box
        cv2.putText(frame, class_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Class label

    return frame

def process_video(input_video_path, output_video_path, model, device, CLASSES):
    """
    Process a video:
    1. Read the video frame by frame.
    2. Process each frame using the model.
    3. Save or display the output video.
    """
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Process the frame
        processed_frame = process_frame(frame, model, device, CLASSES)

        # Write the processed frame to the output video
        out.write(processed_frame)

        # Display the processed frame (optional)
        cv2.imshow('Processed Frame', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
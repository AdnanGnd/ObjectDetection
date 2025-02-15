import cv2
import numpy as np
from torchvision.transforms import ToTensor
import torch

def process_frame(frame, model, device, CLASSES):
    # Preprocess the frame
    input_size = (224, 224)  # Match the input size expected by the model
    original_height, original_width, _ = frame.shape

    # Resize the frame while preserving the aspect ratio
    scale = min(input_size[0] / original_width, input_size[1] / original_height)
    resized_width = int(original_width * scale)
    resized_height = int(original_height * scale)
    resized_frame = cv2.resize(frame, (resized_width, resized_height))

    # Pad the resized frame to match the input size
    pad_width = input_size[0] - resized_width
    pad_height = input_size[1] - resized_height
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    padded_frame = cv2.copyMakeBorder(resized_frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Convert to tensor and add batch dimension
    frame_tensor = ToTensor()(padded_frame).unsqueeze(0).to(device)

    # Run the model
    with torch.no_grad():
        class_output, box_output = model(frame_tensor)

    # Convert predictions to class labels and bounding boxes
    class_ids = torch.argmax(class_output, dim=1).cpu().numpy()  # Predicted classes
    bboxes = box_output.cpu().numpy()  # Bounding box coordinates [x_min, y_min, x_max, y_max]

    # Scale bounding box coordinates back to the original frame size
    bboxes[:, 0] = (bboxes[:, 0] * input_size[0] - pad_left) / scale  # x_min
    bboxes[:, 1] = (bboxes[:, 1] * input_size[1] - pad_top) / scale  # y_min
    bboxes[:, 2] = (bboxes[:, 2] * input_size[0] - pad_left) / scale  # x_max
    bboxes[:, 3] = (bboxes[:, 3] * input_size[1] - pad_top) / scale  # y_max
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
        raise Exception(f"Error: Could not open video {input_video_path}")

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

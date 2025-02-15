import torch

# Load the trained model
model = ObjectDetectionCNN(num_classes=len(CLASSES))  # Replace with your model class
model.load_state_dict(torch.load('best_model.pth'))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Define paths
input_video_path = 'path/to/input_video.mp4'
output_video_path = 'path/to/output_video.mp4'

# Process the video
process_video(input_video_path, output_video_path, model, device, CLASSES)
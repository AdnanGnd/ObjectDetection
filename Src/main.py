import torch
from model import ObjectDetectionCNN
from frameProcessing import process_video
import torch

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define CLASSES
CLASSES = ["Motorcycle", "Pedestrian", "Pedestrian-Crossing", "Prohibition-Sign",
           "Red-Traffic-Light", "Speed-Limit-Sign", "Truck", "Warning-Sign"]


# Load the trained model
model = ObjectDetectionCNN(num_classes=len(CLASSES))  # Replace with your model class
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()  # Set the model to evaluation mode


# Define paths
input_video_path = 'path/to/input_video.mp4'
output_video_path = 'path/to/output_video.mp4'

# Process the video
process_video(input_video_path, output_video_path, model, device, CLASSES)
from ultralytics import YOLO
import torch

def main():

    # Check if CUDA (GPU) is available
    if torch.cuda.is_available():
        print("CUDA is available. Training will use the GPU.")
    else:
        print("CUDA is not available. Training will use the CPU.")


    # Load YOLOv8 model (you can replace 'yolov8n.pt' with any specific YOLOv8 model)
    model = YOLO('yolo11n.pt')  

    # Train the model
    model.train(
        data='../project_dataset/road_detection/road_detection/config.yaml',            # Path to the dataset config YAML
        epochs=50,                     # Number of training epochs
        imgsz=416,                     # Input image size (set to 416 as per your YAML)
        batch=16,                      # Batch size
        optimizer='AdamW',             # Using AdamW as per your original request
        lr0=0.005,                     # Initial learning rate (set as per your original YAML)
        lrf=0.01,                      # Final learning rate factor
        momentum=0.937,                # Momentum for SGD (though using AdamW, this might not be necessary)
        weight_decay=0.0001,           # Weight decay (set as per your original YAML)
        patience=10,                   # Early stopping patience
        amp=True,                      # Automatic mixed precision (for faster training with GPUs)
        project='params/',             # Folder to save the training results
        name='road_detection',         # Name of the run for organizing results
        workers=16,                    # Number of workers for data loading
        device=0,                      # Use GPU 0, or change if using multiple GPUs
    )

if __name__ == '__main__':
    main()

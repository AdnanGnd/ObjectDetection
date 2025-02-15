# Object Detection Project

This project implements an object detection model using a custom dataset with YOLO-format labels.

## Project Structure

- `dataset.py`: Contains the `CustomObjectDetectionDataset` class for loading and processing the dataset.
- `model.py`: Defines the `ObjectDetectionCNN` model architecture.
- `train.py`: Implements the training loop and logic.
- `config.yaml`: Configuration file for setting parameters.
- `utils.py`: Utility functions (e.g., `yolo_to_bbox`).
- `requirements.txt`: Lists all the dependencies required for the project.
- `README.md`: This file.

## Installation

1. Clone the repository.
2. Install the dependencies using `pip install -r requirements.txt`.

## Usage

1. Update the `config.yaml` file with the appropriate paths and parameters.
2. Run the training script: `python train.py`.



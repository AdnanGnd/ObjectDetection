import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision.transforms import ToTensor

def yolo_to_bbox(yolo_bbox, image_width, image_height):
    """
    Convert YOLO format [class_id, x_center, y_center, width, height] to [x_min, y_min, x_max, y_max].
    """
    class_id, x_center, y_center, width, height = yolo_bbox
    x_min = (x_center - width / 2) * image_width
    y_min = (y_center - height / 2) * image_height
    x_max = (x_center + width / 2) * image_width
    y_max = (y_center + height / 2) * image_height
    return [x_min, y_min, x_max, y_max]

class CustomObjectDetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the directory with images.
            label_dir (str): Path to the directory with YOLO-format labels.
            transform (callable, optional): Optional transform to be applied to the image.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # Load label
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace('.jpg', '.txt'))
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # Parse YOLO-format labels and convert to [x_min, y_min, x_max, y_max]
        boxes = []
        labels = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width_norm = float(parts[3])
            height_norm = float(parts[4])

            # Convert YOLO format to [x_min, y_min, x_max, y_max]
            bbox = yolo_to_bbox([class_id, x_center, y_center, width_norm, height_norm], width, height)
            boxes.append(bbox)
            labels.append(class_id)  # Store class_id

        # Apply transforms (if any)
        if self.transform:
            image = self.transform(image)

        # Convert to tensors
        image = ToTensor()(image)  # Convert image to tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)  # Bounding boxes tensor
        labels = torch.tensor(labels, dtype=torch.int64)  # Class labels tensor

        return image, boxes, labels
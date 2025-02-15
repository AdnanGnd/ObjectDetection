import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import yaml
from dataset import CustomObjectDetectionDataset
from model import ObjectDetectionCNN

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create datasets and data loaders
train_dataset = CustomObjectDetectionDataset(config['train_image_dir'], config['train_label_dir'], transform=ToTensor())
val_dataset = CustomObjectDetectionDataset(config['val_image_dir'], config['val_label_dir'], transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

# Initialize model, loss functions, and optimizer
model = ObjectDetectionCNN(num_classes=config['num_classes']).to(device)
classification_criterion = nn.CrossEntropyLoss()  # Multi-class classification
regression_criterion = nn.SmoothL1Loss()  # Bounding box regression
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# Training loop
num_epochs = config['num_epochs']
best_val_loss = float('inf')  # Track the best validation loss

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, boxes, labels in train_loader:
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        # Forward pass
        class_output, box_output = model(images)

        # Compute losses
        classification_loss = classification_criterion(class_output, labels)
        regression_loss = regression_criterion(box_output, boxes)
        total_loss = classification_loss + regression_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()

    # Print training loss
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / len(train_loader)}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, boxes, labels in val_loader:
            images = images.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)

            # Forward pass
            class_output, box_output = model(images)

            # Compute losses
            classification_loss = classification_criterion(class_output, labels)
            regression_loss = regression_criterion(box_output, boxes)
            total_loss = classification_loss + regression_loss

            val_loss += total_loss.item()

    # Print validation loss
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss}")

    # Save the model if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Saved best model with Val Loss: {val_loss}")
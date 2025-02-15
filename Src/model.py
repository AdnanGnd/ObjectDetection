import torch
import torch.nn as nn

class ObjectDetectionCNN(nn.Module):
    def __init__(self, num_classes):
        super(ObjectDetectionCNN, self).__init__()

        # Backbone (Feature Extraction)
        self.backbone = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classification Head (Multi-class)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),  # Output: num_classes
        )

        # Regression Head (Bounding Box Coordinates)
        self.regressor = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4),  # Output: 4 values for [x_min, y_min, x_max, y_max]
            nn.Sigmoid()  # Normalize coordinates to [0, 1]
        )

    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten for fully connected layers

        # Classification output
        class_output = self.classifier(features)

        # Regression output
        box_output = self.regressor(features)

        return class_output, box_output
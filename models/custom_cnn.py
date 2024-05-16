from typing import List
from torch import nn

class CustomCNN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 labels: List[str],
                 labels_classes: int,
                 dropout_p: float,
                 device: any
                ):
        super(CustomCNN, self).__init__()
        self.in_channels = in_channels
        self.labels = labels
        self.labels_classes = labels_classes
        self.dropout_p = dropout_p
        self.device = device

        # Define the layers of your CNN
        self.dropout = nn.Dropout(p=self.dropout_p) # Dropout layer
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        # Fully connected layers
        self.fc1 = nn.Linear(3072, 256)
        self.fc2 = nn.Linear(256, len(self.labels))

    def forward(self, x):
        x = x.to(self.device) # Move data to device

        # Apply convolutional layers with ReLU activation and dropout
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
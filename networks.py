import torch
import torch.nn as nn
import cv2
import numpy as np

class MyNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(
            in_channels=1,
            out_channels=10,
            kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(
            in_channels=10,
            out_channels=20,
            kernel_size=5,
            ),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=20 * 4 * 4, out_features=50),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return nn.functional.log_softmax(x, dim=1)

class GaborLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(GaborLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Initialize Gabor filters as fixed parameters
        self.filters = nn.Parameter(self.create_gabor_filters(), requires_grad=False)
        
    def create_gabor_filters(self):
        num_orientations = self.out_channels // self.in_channels
        gabor_filters = []
        
        for i in range(self.in_channels):
            for j in range(num_orientations):
                theta = j * np.pi / num_orientations  # Orientation
                kernel = cv2.getGaborKernel((self.kernel_size, self.kernel_size),
                                            sigma=2.0,
                                            theta=theta,
                                            lambd=10.0,
                                            gamma=0.5,
                                            psi=0,
                                            ktype=cv2.CV_32F)
                gabor_filters.append(kernel)
        
        # Stack filters and convert to a PyTorch tensor
        filters_tensor = torch.tensor(np.array(gabor_filters), dtype=torch.float32)
        filters_tensor = filters_tensor.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        
        return filters_tensor

    def forward(self, x):
        x = nn.functional.conv2d(x, self.filters, padding=self.kernel_size//2)
        return x

class MyGaborNetwork(nn.Module):
    def __init__(self, input_size=(1, 28, 28)):
        super(MyGaborNetwork, self).__init__()
        self.block_1 = GaborLayer(in_channels=1, out_channels=10, kernel_size=5)
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )

        # Dynamically calculate the input features to the linear layer
        with torch.no_grad():
            self.example_input = torch.zeros(1, *input_size)
            self.example_output = self.forward_features(self.example_input)
            in_features = self.example_output.numel()
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 50),
            nn.ReLU(),  # Added ReLU here for better network performance
            nn.Linear(50, 10)
        )

    def forward_features(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)
        return nn.functional.log_softmax(x, dim=1)

import torch
import torch.nn as nn
import cv2
import numpy as np

class MyNetwork(nn.Module):
    """
    A custom neural network architecture for image classification.

    This network consists of two convolutional blocks followed by a classifier. Each convolutional block
    applies a convolution operation, followed by max pooling and a ReLU activation function. The first block
    moves from 1 input channel to 10 output channels, and the second block increases this to 20 output channels.
    A dropout layer is also included in the second block to reduce overfitting. The classifier is a sequence of
    fully connected layers that flattens the output of the convolutional blocks and then applies linear transformations
    to achieve the final class predictions.

    Attributes:
    - block_1 (nn.Sequential): First convolutional block.
    - block_2 (nn.Sequential): Second convolutional block with dropout.
    - classifier (nn.Sequential): Classifier that flattens and transforms the output into class predictions.
    """

    def __init__(self):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5),
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
        """
        Defines the forward pass of the network.

        Parameters:
        - x (Tensor): The input tensor.

        Returns:
        - Tensor: The output of the network after applying log softmax on the classifier's output.
        """
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return nn.functional.log_softmax(x, dim=1)

class GaborLayer(nn.Module):
    """
    A convolutional layer using Gabor filters as fixed parameters.

    This layer initializes Gabor filters based on the specified number of input and output channels and the kernel size.
    The filters are used in a fixed convolution operation that does not update during training, simulating the effect of
    Gabor filters on the input image.

    Attributes:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - kernel_size (int): Size of the convolution kernel.
    - filters (Tensor): The initialized Gabor filters.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(GaborLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Initialize Gabor filters as fixed parameters
        self.filters = nn.Parameter(self.create_gabor_filters(), requires_grad=False)
        
    def create_gabor_filters(self):
        """
        Creates Gabor filters for convolution.

        Returns:
        - Tensor: A tensor containing the initialized Gabor filters.
        """
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
        """
        Applies Gabor filters to the input.

        Parameters:
        - x (Tensor): The input tensor.

        Returns:
        - Tensor: The output tensor after applying the Gabor filters.
        """
        x = nn.functional.conv2d(x, self.filters, padding=self.kernel_size//2)
        return x

class MyGaborNetwork(nn.Module):
    """
    A custom neural network that incorporates Gabor filters at the first layer.

    This network uses a GaborLayer as its first block to process input images, aiming to capture texture
    and orientation features efficiently. Following the GaborLayer, a conventional convolutional block
    applies further transformations. The network dynamically calculates the necessary input features for
    the linear layer based on a dummy forward pass, ensuring compatibility with various input sizes.

    Attributes:
    - block_1 (GaborLayer): The initial layer using Gabor filters.
    - block_2 (nn.Sequential): A standard convolutional block including dropout, max pooling, and ReLU activation.
    - classifier (nn.Sequential): A classifier that includes a fully connected network with a ReLU activation.
    """

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
            nn.ReLU(),  # Added ReLU for improved performance
            nn.Linear(50, 10)
        )

    def forward_features(self, x):
        """
        Processes input through the initial blocks to determine feature size for the classifier.

        Parameters:
        - x (Tensor): The input tensor.

        Returns:
        - Tensor: The output tensor from the initial processing blocks.
        """
        x = self.block_1(x)
        x = self.block_2(x)
        return x

    def forward(self, x):
        """
        Defines the forward pass of the network with Gabor filters.

        Parameters:
        - x (Tensor): The input tensor.

        Returns:
        - Tensor: The output of the network after applying log softmax on the classifier's output.
        """
        x = self.forward_features(x)
        x = self.classifier(x)
        return nn.functional.log_softmax(x, dim=1)

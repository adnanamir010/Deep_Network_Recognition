import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from networks import MyNetwork
import cv2
import numpy as np

"""
    Train the network for one epoch.
    
    This function iterates over the training dataset using the provided DataLoader,
    computes the loss for each batch, and updates the model parameters. It tracks and
    returns the average training loss for the epoch.
    
    Parameters:
    - model: The neural network model to be trained.
    - device: The computing device (CPU or GPU) where the training is performed.
    - train_loader (DataLoader): DataLoader for the training dataset.
    - optimizer: The optimization algorithm used to update the model parameters.
    - epoch (int): The current training epoch number.
    
    Returns:
    - average_loss (float): The average loss over the training dataset for the epoch.
    - total_data (int): Total number of samples in the training dataset.
"""
def train_network(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch}', leave=False)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_loader)
    return average_loss, len(train_loader.dataset)


"""
    Evaluate the network on the test dataset.

    This function iterates over the test dataset, computes the loss for each batch,
    and calculates the overall test loss and accuracy of the model. It sets the model
    to evaluation mode before testing, disables gradient computations, and prints out
    the average test loss and accuracy.

    Parameters:
    - model: The neural network model to be evaluated.
    - device: The computing device (CPU or GPU) where the evaluation is performed.
    - test_loader (DataLoader): DataLoader for the test dataset.

    Returns:
    - test_loss (float): The average loss over the test dataset.
    - test_accuracy (float): The accuracy of the model on the test dataset.
"""
def test_network(model, device, test_loader):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing', leave=True)
    with torch.no_grad():
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            # Optionally, you can set the post-fix to display additional information
            pbar.set_postfix(loss=test_loss/(batch_idx+1), accuracy=100. * correct/len(test_loader.dataset))
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    
    print(f'Average test loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')

    return test_loss, test_accuracy

"""
    Process an input image and predict the output using a specified model.

    This function reads an image from a given path in grayscale, optionally inverts it,
    resizes it to 28x28 pixels, normalizes its pixel values to the range [0, 1], applies
    MNIST-specific normalization, and then passes the processed image tensor through the
    given model to get the prediction.

    Parameters:
    - image_path (str): The file path to the image to process.
    - model: The prediction model to use.
    - device: The device (CPU or GPU) to use for computations.
    - invert (bool): If True, invert the colors of the image.

    Returns:
    - The output from the model prediction.
"""
def process_and_predict(image_path, model, device,invert):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
    if invert:
        image = cv2.bitwise_not(image)
    image = cv2.resize(image, (28, 28))  # Resize to 28x28
    image = image.astype(np.float32) / 255.0  # Normalize image pixels to [0, 1]
    tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions
    tensor = (tensor - 0.1307) / 0.3081  # Apply MNIST normalization
    output = model(tensor) # Run inference/prediction
    return output

"""
    A transformation class for preprocessing greek alphabet images provided with this course . This class applies
    a series of transformations to convert an input image to grayscale, scale it
    maintaining aspect ratio, perform a center crop, and finally invert its colors.

    The sequence of transformations is as follows:
    1. Convert the image to grayscale to focus on shape and texture rather than color.
    2. Scale the image to a specific ratio (simulating the aspect ratio found in Greek art) without rotation.
    3. Center crop the image to a fixed size (28x28), highlighting the central figure or motif.
    4. Invert the image colors, creating a visual style reminiscent of ancient Greek pottery.

"""
class GreekTransform:
    def __call__(self, x):
        x = transforms.functional.rgb_to_grayscale(x)
        x = transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = transforms.functional.center_crop(x, (28, 28))
        return transforms.functional.invert(x)
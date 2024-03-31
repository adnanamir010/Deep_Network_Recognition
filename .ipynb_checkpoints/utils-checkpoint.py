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

def visualize_data(test_loader,nums):
    dataiter = iter(test_loader)
    images, labels = next(dataiter)  # Corrected usage
    images = images.numpy()

    fig = plt.figure(figsize=(9, 3))
    for idx in range(nums):
        ax = fig.add_subplot(1, nums, idx+1, xticks=[], yticks=[])
        ax.imshow(images[idx].squeeze(), cmap='gray')
        ax.set_title(str(labels[idx].item()))
    plt.show()
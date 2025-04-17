from model import train_data, train_dataloader, valid_dataloader, test_dataloader, train_epoch, evaluate_epoch, train, plot_result 
from BaseModel import base_metrics

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

np.random.seed(seed=941)
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

x = np.random.randint(low=0, high=255, size=(3, 3)).astype(np.float64)
x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# class SimpleResBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(SimpleResBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(in_channels)
#         self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(in_channels)

#     def forward(self, x):
#         identity = x
#         print('1', identity)
#         out = self.conv1(x)
#         print('2', out)
#         out = self.bn1(out)
#         print('3', out)
#         out = F.relu(out)
#         print('4', out)
#         out = self.conv2(out)
#         print('5', out)
#         out = self.bn2(out)
#         print('6', out)
#         out += identity
#         print('7', out)
#         out = F.relu(out)

#         return out
    
# in_channels = 1
# skip_connection_model = SimpleResBlock(in_channels)
# skip_connection_model(x)

class SimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity

        out = F.relu(out)

        return out
    
class SimpleNet(nn.Module):
    def __init__(self, in_channels, num_classes, num_blocks, block_channels):
        super(SimpleNet, self).__init__()
        layers = []
        for i in range(num_blocks):
            layers.append(SimpleBlock(in_channels if i == 0 else block_channels, block_channels))
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(block_channels, num_classes)  # Fully connected layer

    def forward(self, x):
        out = self.blocks(x)

        # Flattening and fully connected layer
        out = F.adaptive_avg_pool2d(out, (1, 1))  # Adaptive pooling to size (1, 1)
        out = torch.flatten(out, 1)  # Flatten the tensor
        out = self.fc(out)  # Fully connected layer

        return out
    
in_channels = 1
skip_connection_model = SimpleNet(in_channels, num_classes=len(train_data.dataset.classes), num_blocks=4, block_channels=32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
skip_connection_model = skip_connection_model.to(device)
# summary(skip_connection_model, (1, 28, 28))

# Train skip_connection_model
num_classes = len(train_data.dataset.classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
skip_connection_model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(skip_connection_model.parameters())
num_epochs = 20
save_model = './model'
os.makedirs(save_model, exist_ok = True)
model_name = 'skip_connection_model'
skip_connection_model, base_metrics = train(
    skip_connection_model, model_name, optimizer, criterion, train_dataloader, valid_dataloader, num_epochs, device
)

# Calculate test_acc and test_loss of skip_connection_model
test_acc, test_loss = evaluate_epoch(skip_connection_model, criterion, test_dataloader)
test_acc, test_loss

# Plot Result
plot_result(
    num_epochs,
    base_metrics["train_accuracy"],
    base_metrics["valid_accuracy"],
    base_metrics["train_loss"],
    base_metrics["valid_loss"]
)
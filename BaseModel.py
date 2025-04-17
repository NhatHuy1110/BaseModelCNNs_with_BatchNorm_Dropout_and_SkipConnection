from model import train_data, train_dataloader, valid_dataloader, test_dataloader, train_epoch, evaluate_epoch, train, plot_result 

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

base_model = nn.Sequential(
    nn.Conv2d(1, 6, 5, stride=2),
    nn.Flatten(),
    nn.Linear(6 * 12 * 12, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)

# summary(base_model, (1, 28, 28))

# Train base_model
num_classes = len(train_data.dataset.classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(base_model.parameters())
num_epochs = 20
save_model = './model'
os.makedirs(save_model, exist_ok = True)
model_name = 'base_model'
base_model, base_metrics = train(
    base_model, model_name, optimizer, criterion, train_dataloader, valid_dataloader, num_epochs, device
)

# Calculate test_acc and test_loss of base_model
test_acc, test_loss = evaluate_epoch(base_model, criterion, test_dataloader)
test_acc, test_loss

# Plot Result 
plot_result(
    num_epochs,
    base_metrics["train_accuracy"],
    base_metrics["valid_accuracy"],
    base_metrics["train_loss"],
    base_metrics["valid_loss"]
)
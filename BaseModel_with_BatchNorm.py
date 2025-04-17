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

batchnorm_model = nn.Sequential(
    nn.Conv2d(1, 6, 5, stride=2),
    nn.Flatten(),
    nn.Linear(6*12*12, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.Linear(32, 10)
)

# summary(batchnorm_model, (1, 28, 28))

# Train batchnorm_model
num_classes = len(train_data.dataset.classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batchnorm_model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(batchnorm_model.parameters())
num_epochs = 20
save_model = './model'
os.makedirs(save_model, exist_ok = True)
model_name = 'batch_norm_model'
batchnorm_model, batchnorm_metrics = train(
    batchnorm_model, model_name, optimizer, criterion, train_dataloader, valid_dataloader, num_epochs, device
)

# Calculate test_acc and test_loss of batchnorm_model
test_acc, test_loss = evaluate_epoch(batchnorm_model, criterion, test_dataloader)
test_acc, test_loss

# Plot Result 
plot_result(
    num_epochs,
    batchnorm_metrics["train_accuracy"],
    batchnorm_metrics["valid_accuracy"],
    batchnorm_metrics["train_loss"],
    batchnorm_metrics["valid_loss"]
)

# Plot to compare between with BatchNorm and without BatchNorm
epochs = list(range(num_epochs))
fig, axs = plt.subplots(nrows = 1, ncols =2 , figsize = (12,6))
axs[0].plot(epochs, base_metrics['train_loss'], label = "w/o BN")
axs[0].plot(epochs, batchnorm_metrics['train_loss'], label = "w BN")
axs[1].plot(epochs, base_metrics['valid_loss'], label = "w/o BN")
axs[1].plot(epochs, batchnorm_metrics['valid_loss'], label = "w BN")
axs[0].set_xlabel("Epochs")
axs[1].set_xlabel("Epochs")
axs[0].set_ylabel("Training Loss")
axs[1].set_ylabel("Evaluation Loss")
plt.legend()
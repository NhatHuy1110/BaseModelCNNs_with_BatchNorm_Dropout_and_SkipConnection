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

dropout_model = nn.Sequential(
    nn.Conv2d(1, 6, 5, stride=2),
    nn.Flatten(),
    nn.Linear(6 * 12 * 12, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 10)
)

# summary(dropout_model, (1, 28, 28))

# Train dropout_model
num_classes = len(train_data.dataset.classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dropout_model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(dropout_model.parameters())
num_epochs = 20
save_model = './model'
os.makedirs(save_model, exist_ok = True)
model_name = 'dropout_model'
dropout_model, dropout_metrics = train(
    dropout_model, model_name, optimizer, criterion, train_dataloader, valid_dataloader, num_epochs, device
)

# Calculate test_acc and test_loss of dropout_model
test_acc, test_loss = evaluate_epoch(dropout_model, criterion, test_dataloader)
test_acc, test_loss

# Plot Result 
plot_result(
    num_epochs,
    dropout_metrics["train_accuracy"],
    dropout_metrics["valid_accuracy"],
    dropout_metrics["train_loss"],
    dropout_metrics["valid_loss"]
)

# Plot to compare between with Dropout and without Dropout
epochs = list(range(num_epochs))
fig, axs = plt.subplots(nrows = 1, ncols =2 , figsize = (12,6))
axs[0].plot(epochs, base_metrics['train_loss'], label = "w/o Dropout")
axs[0].plot(epochs, dropout_metrics['train_loss'], label = "w Dropout")
axs[1].plot(epochs, base_metrics['valid_loss'], label = "w/o Dropout")
axs[1].plot(epochs, dropout_metrics['valid_loss'], label = "w Dropout")
axs[0].set_xlabel("Epochs")
axs[1].set_xlabel("Epochs")
axs[0].set_ylabel("Training Loss")
axs[1].set_ylabel("Evaluation Loss")
plt.legend()

# Plot to show the Evaluation Loss and Evaluation Accuracy with every dropout p
metrics = []
p_dropouts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for p in p_dropouts:
    print(p)
    dropout_model = nn.Sequential(
        nn.Conv2d(1, 6, 5, stride=2),
        nn.Flatten(),
        nn.Linear(6 * 12 * 12, 64),
        nn.ReLU(),
        nn.Dropout(p),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(p),
        nn.Linear(32, 10)
    )
    dropout_model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(dropout_model.parameters())
    num_epochs = 20
    save_model = './model'
    os.makedirs(save_model, exist_ok = True)
    model_name = 'dropout_model'
    dropout_model, dropout_metrics = train(
        dropout_model, model_name, optimizer, criterion, train_dataloader, valid_dataloader, num_epochs, device
    )
    test_acc, test_loss = evaluate_epoch(dropout_model, criterion, test_dataloader)
    dropout_metrics['test_accuracy'] = test_acc
    dropout_metrics['test_loss'] = test_loss
    metrics.append(dropout_metrics)

epochs = list(range(num_epochs))
fig, axs = plt.subplots(nrows = 1, ncols =2 , figsize = (12,6))

for idx, metric in enumerate(metrics):
    axs[0].plot(epochs, metric['valid_loss'], label=p_dropouts[idx])
    axs[1].plot(epochs, metric['valid_accuracy'], label=p_dropouts[idx])
axs[0].set_xlabel("P")
axs[1].set_xlabel("P")
axs[0].set_ylabel("Evaluation Loss")
axs[1].set_ylabel("Evaluation Accuracy")
plt.legend()

plt.plot(p_dropouts, [sum(metric['time'])/len(metric['time']) for metric in metrics])
plt.show()
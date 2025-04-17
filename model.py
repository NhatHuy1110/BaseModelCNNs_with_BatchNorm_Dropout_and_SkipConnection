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

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic=True

#######################################################
##                    Load Dataset                   ##
#######################################################

ROOT = './data'

train_data = datasets.MNIST(
    root=ROOT,
    train=True, 
    download=True
)

test_data = datasets.MNIST(
    root=ROOT,
    train=True,
    download=True
)

#######################################################
##                    PreProcessing                  ##
#######################################################

VALID_RATIO = 0.9 # 90% for training and 10% for testing

n_train_examples = int(len(train_data) * VALID_RATIO) 
n_valid_examples = len(train_data) - n_train_examples

train_data, valid_data = data.random_split(
    train_data,
    [n_train_examples, n_valid_examples]
)
# compute mean and variance(std)
# divide for 255 to normalize them to [0, 1]
mean = train_data.dataset.data.float().mean()/255 
std = train_data.dataset.data.float().std()/255

train_transforms = transforms.Compose([
    transforms.ToTensor(),                      # Convert numpy arrays to tensor
    transforms.Normalize(mean=[mean], std=[std])# Normalize them with mean and std
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean], std=[std])
])

train_data.dataset.transforms = train_transforms
valid_data.dataset.transforms = test_transforms
test_data.transforms = test_transforms

BATCH_SIZE = 256
train_dataloader = data.DataLoader(
    train_data,
    shuffle=True,
    batch_size=BATCH_SIZE
)
valid_dataloader = data.DataLoader(
    valid_data,
    batch_size=BATCH_SIZE
)
test_dataloader = data.DataLoader(
    test_data,
    batch_size=BATCH_SIZE
)

#######################################################
##                  Train Functions                  ##
#######################################################

def train_epoch(model, optimizer, criterion, train_dataloader, device, epoch=0, log_interval=50):
    model.train()
    total_acc, total_count = 0, 0
    losses = []
    start_time = time.time()

    for idx, (inputs, labels) in enumerate(train_dataloader):
        inputs=inputs.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()

        predictions = model(inputs)

        # compute loss
        loss = criterion(predictions, labels)
        losses.append(loss.item())

        # backward propagation
        loss.backward()
        optimizer.step()
        total_acc += (predictions.argmax(1) == labels).sum().item()
        total_count += labels.size(0)

        if idx % log_interval == 0 and idx > 0:
            elaped = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(train_dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()
    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_acc, epoch_loss

def evaluate_epoch(model, criterion, valid_dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    losses = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_dataloader):
            inputs=inputs.to(device)
            labels=labels.to(device)

            predictions = model(inputs)

            loss=criterion(predictions, labels)
            losses.append(loss)

            total_acc += (predictions.argmax(1) == labels).sum().item()
            total_count += labels.size(0)

    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_acc, epoch_loss

def train(model, model_name, optimizer, criterion, train_dataloader, valid_dataloader, num_epochs, device):
    train_accs, train_losses =[], []
    eval_accs, eval_losses = [], []
    best_loss_eval = 100
    times = [] 
    for epoch in range(1, num_epochs+1):
        epoch_start_time = time.time()

        #Training
        train_acc, train_loss = train_epoch(model, optimizer, criterion, train_dataloader, device, epoch)
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        #Evaluation
        eval_acc, eval_loss = evaluate_epoch(model, criterion, valid_dataloader)
        eval_accs.append(eval_acc)
        eval_losses.append(eval_loss)

        # Save best model if eval_loss less than 100
        if eval_loss < best_loss_eval:
            torch.save(model.static_dict, save_model + f'/{model_name}.pt')
        
        times.append(time.time() - epoch_start_time)

        # Print loss, acc end epoch
        print("-" * 59)
        print(
            "| End of epoch {:3d} | Time: {:5.2f}s | Train Accuracy {:8.3f} | Train Loss {:8.3f} "
            "| Valid Accuracy {:8.3f} | Valid Loss {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, train_acc, train_loss, eval_acc, eval_loss
            )
        )
        print("-" * 59)

        # Load best model
        model.load_state_dict(torch.load(save_model + f'/{model_name}.pt'))
        model.eval()
    metrics = {
        'train_accuracy': train_accs,
        'train_loss': train_losses,
        'valid_accuracy': eval_accs,
        'valid_loss': eval_losses,
        'time': times
    }
    return model, metrics

#######################################################
##                    Plot Result                    ##
#######################################################

def plot_result(num_epochs, train_accs, eval_accs, train_losses, eval_losses):
    epochs = list(range(num_epochs))
    fig, axs = plt.subplots(nrows = 1, ncols =2 , figsize = (12,6))
    axs[0].plot(epochs, train_accs, label = "Training")
    axs[0].plot(epochs, eval_accs, label = "Evaluation")
    axs[1].plot(epochs, train_losses, label = "Training")
    axs[1].plot(epochs, eval_losses, label = "Evaluation")
    axs[0].set_xlabel("Epochs")
    axs[1].set_xlabel("Epochs")
    axs[0].set_ylabel("Accuracy")
    axs[1].set_ylabel("Loss")
    plt.legend()
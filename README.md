# BaseModelCNNs_with_BatchNorm_Dropout_and_SkipConnection

This project explores and compares the performance of a simple Convolutional Neural Network (CNN) model using BaseModel with different enhancements. Specifically, it compares the following variations of the base model:
- **Base Model**: A simple CNN model with one convolutional layer and fully connected layers.
- **Base Model with Batch Normalization**: Adds BatchNorm layers to the base model to help improve training stability and speed up convergence.
- **Base Model with Dropout**: Adds Dropout layers to the base model to reduce overfitting.
- **Base Model with Skip Connections**: Incorporates residual (skip) connections to improve model performance by allowing gradients to flow more easily during backpropagation.

The primary objective of this project is to analyze how these enhancements (BatchNorm, Dropout, and SkipConnection) affect the performance of the CNN model in terms of accuracy and loss during training and evaluation.

## File Structure

AdvancedCNNs_with_BatchNorm_Dropout_and_SkipConnection/

├── AdvancedCNNs_with_BatchNorm_Dropout_and_SkipConnection.ipynb

├── MNIST dataset.jpg

├── BaseModel.py

├── BaseModel_with_BatchNorm.py

├── BaseModel_with_Dropout.py

├── BaseModel_with_SkipConnection.py

├── model.py

└── README.md

- `AdvancedCNNs_with_BatchNorm_Dropout_and_SkipConnection.ipynb`:  A jupyter notebook that trains and compares these models, showing the results and visualizing the performance.
- `MNIST dataset.jpg`: The MNIST dataset (Images: 70.000, Class: 10, Image Size: 28 x 28).
- `BaseModel.py`: Defines the architecture for the basic CNN model.
- `BaseModel_with_BatchNorm.py`: Implements the model with Batch Normalization layers.
- `BaseModel_with_Dropout.py`: Implements the model with Dropout layers.
- `BaseModel_with_SkipConnection.py`: Implements the model with Skip Connections (residual blocks).
- `model.py`: Contains the dataset loading, data preprocessing, and other utility functions.

# 🔴Note: I divided this project into 2 types of files: .ipynb and .py because if you want to run on ggcolab🚀 to observe the results as my codeflow, you can download file .ipynb and run it, ✅but if you want to have an overview of Model CNNs regular or with BatchNorm, Dropout, and SkipConnection, you can check my .py files

## Implementation

**model.py**: Data Loading, Preprocessing, and Model Functions.
-> In this file, I handle essential tasks for loading and preparing the dataset, defining transformations for data processing, and includes some important training functions like: train_epoch for training train_data, evaluate_epoch for training valid_data, and train for training the model.

```python
def train_epoch(model, optimizer, criterion, train_dataloader, device, epoch=0, log_interval=50):
    ...
##
def evaluate_epoch(model, criterion, valid_dataloader):
    ...
##
def train(model, model_name, optimizer, criterion, train_dataloader, valid_dataloader, num_epochs, device):
    ...
```

**BaseModel.py**: Basic CNN architecture.
-> In this file, I define the basic CNN architecture for the model without any additional enhancements like BatchNorm, Dropout, and SkipConnection.

```python
base_model = nn.Sequential(
    nn.Conv2d(1, 6, 5, stride=2),
    nn.Flatten(),
    nn.Linear(6 * 12 * 12, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)
```

**BaseModel_with_BatchNorm.py**: Base Model with Batch Normalization.
-> In this file, I try to add Batch Normalization layers after the fully connected layers to improve the model's training stability and convergence speed.

```python
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
```

- **BaseModel_with_Dropout.py**: Base Model with Dropout
-> In this file, I try to add Dropout layers to reduce overfitting, especially useful when the model has a large number of parameters.

```python
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
```

- **BaseModel_with_SkipConnection.py**: Base Model with Skip Connections
-> In this file, I try to add Skip Connections (residual blocks) to improve model training and performance by enabling better gradient flow.

```python
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
        self.fc = nn.Linear(block_channels, num_classes)

    def forward(self, x):
        out = self.blocks(x)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

skip_connection_model = SimpleNet(in_channels, num_classes=len(train_data.dataset.classes), num_blocks=4, block_channels=32)
```

## Result 

| Model                           | Test Accuracy |
|---------------------------------|---------------|
| CNN basic                       |   ~ 97,53%    |
| CNN using BatchNorm Layer       |   ~ 97,85%    |
| CNN using Dropout Layer         |   ~ 97,58%    | 
| CNN using SkipConnection Layer  |   ~ 98,78%    |

# 🔴Note: All images illustrating the data of 4 CNNs Model in the AdvancedCNNs_with_BatchNorm_Dropout_and_SkipConnection.ipynb file, or you can run to test by yourself

### Conclusion: 
- Batch Normalization significantly improves convergence and stabilizes training, leading to better generalization on unseen data.

- Dropout is an effective regularization technique that helps reduce overfitting, especially in models with a large number of parameters.

- Skip Connections improve model training efficiency and allow the model to learn more complex features without sacrificing accuracy or generalization.


## Requirement

✅ You can run code on googlecollab, jupyter or Kaggle notebooks, ... following my "AdvancedCNNs_with_BatchNorm_Dropout_and_SkipConnection.ipynb" file
✅ If you don't have a virtual enviroment, manually include the nessesary libraries. Install the necessary dependencies for this project, follow these steps:

    1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/NhatHuy1110/BaseModelCNNs_with_BatchNorm_Dropout_and_SkipConnection.git
    ```

    2. Navigate into the project directory:
    ```bash
    cd BaseModelCNNs_with_BatchNorm_Dropout_and_SkipConnection
    ```

    3. Install the required dependencies by running the following command:
    ```bash
    pip install -r requirements.txt
    ```

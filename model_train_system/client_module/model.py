import torch.nn as nn
import torch.nn.functional as F
from torch import optim


def get_model(model_name="mnist_2nn"):
    if model_name == "mnist_cnn":
        return MNIST_CNN()
    elif model_name == "mnist_2nn":
        return MNIST_2NN()
    elif model_name == "emnist_cnn":
        return EMNIST_CNN()
    elif model_name == "emnist_2nn":
        return EMNIST_2NN()
    else:
        raise ValueError("no model match")


def get_opti(model, learning_rate, opti_name="SGD"):
    if opti_name == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif opti_name == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate)


def get_loss_fn(loss_fn_name="default"):
    return nn.CrossEntropyLoss()


class MNIST_2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 150)
        self.fc2 = nn.Linear(150, 150)
        self.fc3 = nn.Linear(150, 10)

    def forward(self, inputs):
        tensor = F.relu(self.fc1(inputs))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.fc3(tensor)
        return tensor


class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7 * 7 * 48, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7 * 7 * 48)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor


class EMNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7 * 7 * 48, 256)
        self.fc2 = nn.Linear(256, 37)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7 * 7 * 48)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor


class EMNIST_2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 250)
        self.fc2 = nn.Linear(250, 150)
        self.fc3 = nn.Linear(150, 37)

    def forward(self, inputs):
        tensor = F.relu(self.fc1(inputs))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.fc3(tensor)
        return tensor


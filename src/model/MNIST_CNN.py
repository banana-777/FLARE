# 04 29
# 定义模型
# model structure : Conv3*3 MaxPool2*2 Conv3*3 MaxPool2*2 Dropout0.5 Dense128 Dense10
# data : MNIST

import torch
import torch.nn as nn
import torch.optim as optim

class Model_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict({
            'conv1': nn.Conv2d(1, 32, kernel_size=3, padding=1),
            'conv1_relu': nn.ReLU(),
            'pool1': nn.MaxPool2d(2, 2),
            'conv2': nn.Conv2d(32, 64, kernel_size=3, padding=1),
            'conv2_relu': nn.ReLU(),
            'pool2': nn.MaxPool2d(2, 2),
            'dropout': nn.Dropout(0.5),
            'fc1': nn.Linear(7 * 7 * 64, 128),
            'fc1_relu': nn.ReLU(),
            'fc2': nn.Linear(128, 10)
        })

    def forward(self, x):
        x = self.layers['conv1'](x)
        x = self.layers['conv1_relu'](x)
        x = self.layers['pool1'](x)
        x = self.layers['conv2'](x)
        x = self.layers['conv2_relu'](x)
        x = self.layers['pool2'](x)
        x = x.view(-1, 7 * 7 * 64)
        x = self.layers['dropout'](x)
        x = self.layers['fc1'](x)
        x = self.layers['fc1_relu'](x)
        x = self.layers['fc2'](x)
        return x


model = Model_CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

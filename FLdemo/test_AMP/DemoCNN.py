# 05 02
# 定义模型
# model structure : Conv3*3 MaxPool2*2 Conv3*3 MaxPool2*2 Dropout0.5 Dense128 Dense10
# data : MNIST

import torch
import torch.nn as nn
import torch.optim as optim

class Model_CNN(nn.Module):
    def __init__(self):
        super(Model_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # 输出28x28x32
        self.pool = nn.MaxPool2d(2, 2)                          # 输出14x14x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)# 输出14x14x64
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(7*7*64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, 7*7*64)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型与优化器
model = Model_CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

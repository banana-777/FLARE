# 该文件用于测试模型在划分好的收据集上能否正常执行训练
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

# 设备配置
device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 自定义数据集类
class MNISTClientDataset(Dataset):
    def __init__(self, data_path):
        """
        Args:
            data_path (str): 数据文件路径 (如 ./mnist_data/client_0_train.pt)
        """
        data = torch.load(data_path)
        self.images = data['images']
        self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# CNN模型定义
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 7x7
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        return self.fc_layers(x)


# 训练函数
def train_model(
        train_data_path,
        test_data_path,
        num_epochs=10,
        batch_size=64,
        lr=0.001
):
    # 初始化模型
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 加载数据
    train_dataset = MNISTClientDataset(train_data_path)
    test_dataset = MNISTClientDataset(test_data_path)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        # 计算epoch损失
        epoch_loss = running_loss / len(train_dataset)

        # 每个epoch结束后测试
        test_acc = test_model(model, test_loader)

        print(f'Epoch [{epoch + 1}/{num_epochs}] | '
              f'Loss: {epoch_loss:.4f} | '
              f'Test Acc: {test_acc:.2f}%')

    return model


# 测试函数
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


if __name__ == "__main__":
    # 示例用法：训练客户端0的数据
    client_id = 1
    model = train_model(
        train_data_path=f"./mnist_data/client_{client_id}_train.pt",
        test_data_path=f"./mnist_data/client_{client_id}_test.pt",
        num_epochs=15,
        batch_size=128,
        lr=0.0015
    )

    # 保存模型
    torch.save(model.state_dict(), f"client_{client_id}_model.pt")

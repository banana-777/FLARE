# 04 29
# GUI使用到的库和函数

import torch
from torchvision import transforms
from MNIST_CNN import Model_CNN


# GUI接口类
class FLGUIInterface:
    @staticmethod
    def load_model(model_path, device='cpu'):
        model = Model_CNN()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model

    @staticmethod
    def predict(model, image, device='cpu'):
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
        return output.argmax().item()
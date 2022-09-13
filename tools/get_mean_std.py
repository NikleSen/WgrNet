import torch
import torch.utils.data.dataloader as dataloader
import torchvision.datasets as datasets
from torchvision import transforms


train_dataset = datasets.cifar.CIFAR10(root="./data", train=False, transform=transforms.ToTensor(), download=True)
train_loader = dataloader.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

def get_mean_std(loader):
    #VAR[X]=E[X**2]-E(X)**2
    #公式推导参考https://zhuanlan.zhihu.com/p/35435231
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std

mean, std = get_mean_std(train_loader)
print(mean)
print(std)

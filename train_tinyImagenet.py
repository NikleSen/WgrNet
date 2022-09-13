import os
from datetime import datetime
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from apex import amp
from tools import time2string as t2s
from tools.myseed import seed_torch

# 定义是否使用GPU以及初始化随机数
seed = 1234
seed_torch(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch Tiny Imagenet Training')
parser.add_argument('--model_index', type=int, default=2)
parser.add_argument('--classes', type=int, default=200)
parser.add_argument('--depth_id', type=int, default=0)
parser.add_argument('--width_id', type=int, default=0)
parser.add_argument('--kernel', type=int, default=3)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--expansion', type=float, default=1)
parser.add_argument('--light_id', type=int, default=-1)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--augment', type=bool, default=True)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()
gpu_count = torch.cuda.device_count()

EPOCH = args.epoch
pre_epoch = 0
BATCH_SIZE = args.bs
LR = args.lr
classes = args.classes
depth_id = args.depth_id
width_id = args.width_id
kernel = args.kernel
stride = args.stride
expansion = args.expansion
light_id = args.light_id
model_index = args.model_index
gpu_id = args.gpu_id

if model_index == 2:
    from model.WgrNet import wgrnet

net = wgrnet(3, classes, depth=depth_id, width=width_id, light=light_id, kernel=kernel, stride=stride,
             expansion=expansion)

torch.cuda.set_device(gpu_id)
net.to('cuda')
net_name = type(net).__name__

milestone = [60, 100, 130]  # epoch=160
if EPOCH == 200:
    milestone = [60, 110, 160]  # epoch=200

def worker_init_fn(worker_id):
    random.seed(seed + worker_id)

def tiny_loader(batch_size, data_dir):
    transform_train = transforms.Compose([
        transforms.RandomCrop(size=64,padding=4),
        transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.IMAGENET),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        transforms.RandomErasing(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))])
    trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
    testset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                               worker_init_fn=worker_init_fn)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                              worker_init_fn=worker_init_fn)
    return train_loader, test_loader


trainloader, testloader = tiny_loader(batch_size=BATCH_SIZE, data_dir='/home/hxj/datasets/tiny-imagenet-200')

log_dir = './log_tiny'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
accfile = log_dir + '/TinyImagenet_' + str(net_name) + '_' + str(depth_id) + '_' + str(width_id) + '_' \
          + str(light_id) + '_' + str(kernel) + '_' + str(expansion) + '_' + t2s.get_date_time_str() + '.txt'

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestone, gamma=0.1)

net, optimizer = amp.initialize(net, optimizer, opt_level='O1', verbosity=0)

if __name__ == "__main__":
    best_acc = 0.0
    start = datetime.now()
    print("Start Training, net is: {}".format(net_name))
    with open(accfile, "w") as f:
        f.write(str(args))
        f.write('\n')
        f.write("Epoch,Accuracy,learn_lr")
        f.write('\n')
        f.flush()

        max_acc = 0

        for epoch in range(pre_epoch, EPOCH):
            print('Epoch: %d:' % (epoch + 1))
            net.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            i = 0
            for data in tqdm(trainloader, ncols=70):  # 进度条设置
                optimizer.zero_grad()
                length = len(trainloader)
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)

                loss = criterion(outputs, labels)

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

                optimizer.step()

                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)

                cur_lr = optimizer.param_groups[0]['lr']
                correct += predicted.eq(labels.data).cpu().sum()
            scheduler.step()

            with torch.no_grad():
                correct = 0
                total = 0
                for data in testloader:
                    net.eval()
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                print('Current epoch accuracy = %.3f' % (100 * correct / total), end=', ')
                acc = 100. * correct / total

                f.write("%03d,%.3f,%.6f" % (epoch + 1, acc, cur_lr))
                f.write('\n')
                f.flush()

                if acc > best_acc:
                    best_acc = acc
            if max_acc < best_acc:
                max_acc = best_acc
                max_acc_epoch = epoch + 1
            print("max accuracy= %.3f at [%03d] epoch." % (max_acc, max_acc_epoch))
        f.write("The epoch = %03d,max accuracy= %.3f\n" % (max_acc_epoch, max_acc))
        end = datetime.now()
        h, m, s = t2s.get_h_m_s(end, start)
        f.write("Total running time = %s:%s:%s\n" % (h, m, s))
        f.flush()
        print("Training finished, total epoch = %d." % EPOCH)

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
from tqdm import tqdm
from apex import amp
from tools import time2string as t2s
from tools.myseed import seed_torch

# 定义是否使用GPU以及初始化随机数
seed = 1234
seed_torch(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model_index', type=int, default=0)
parser.add_argument('--classes', type=int, default=10)
parser.add_argument('--depth_id', type=int, default=0)
parser.add_argument('--width_id', type=int, default=0)
parser.add_argument('--kernel', type=int, default=3)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--expansion', type=float, default=0.5)
parser.add_argument('--light_id', type=int, default=-1)
parser.add_argument('--epoch', type=int, default=160)
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--augment', type=bool, default=True)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument('--gpu_id',type=int,default=0)
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
if model_index == 1:
    from model.WgrNet_Group import wgrnet
    net = wgrnet(3, classes, depth=depth_id, width=width_id, light=light_id, kernel=kernel, stride=stride,
                 expansion=expansion)
elif model_index == 2:
    from model.WgrNet import wgrnet
    net = wgrnet(3, classes, depth=depth_id, width=width_id, light=light_id, kernel=kernel, stride=stride,
             expansion=expansion)

torch.cuda.set_device(gpu_id)
net.to('cuda')
net_name = type(net).__name__

milestone = [60, 100, 130]          # epoch=160
if EPOCH == 200:
    milestone = [60, 110, 160]      # epoch=200

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    transforms.RandomErasing(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


def worker_init_fn(worker_id):
    random.seed(seed + worker_id)


trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)  # 训练数据集
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8,
                         worker_init_fn=worker_init_fn)

testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=False, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, pin_memory=True, num_workers=8,
                        worker_init_fn=worker_init_fn)

log_dir = './log_10'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
accfile = log_dir + '/Cifar10_' + str(net_name) +'_'+str(model_index)+ '_' + str(depth_id) + '_' + str(width_id) + '_' \
          + str(light_id) + '_' + str(kernel) + '_' + str(expansion) +'_' + t2s.get_date_time_str() + '.txt'

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

                length = len(trainloader)
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                # loss.backward()

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)

                cur_lr = optimizer.param_groups[0]['lr']
                correct += predicted.eq(labels.data).cpu().sum()
            if amp._amp_state.loss_scalers[0]._unskipped != 0:  # assuming you are using a single optimizer
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
                print('Current epoch accuracy = %.2f' % (100 * correct / total), end=', ')
                acc = 100. * correct / total

                f.write("%03d,%.2f,%.6f" % (epoch + 1, acc, cur_lr))
                f.write('\n')
                f.flush()

                if acc > best_acc:
                    best_acc = acc
            if max_acc < best_acc:
                max_acc = best_acc
                max_acc_epoch = epoch + 1
            print("max accuracy= %.2f at [%03d] epoch." % (max_acc, max_acc_epoch))
        f.write("The epoch = %03d,max accuracy= %.2f\n" % (max_acc_epoch, max_acc))
        end = datetime.now()
        h, m, s = t2s.get_h_m_s(end, start)
        f.write("Total running time = %s:%s:%s\n" % (h, m, s))
        f.flush()
        print("Training finished, total epoch = %d." % EPOCH)

import time
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../dataset_1", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="../dataset_1", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))


# Dataloader装载
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 创建网络模型
class Phd(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5), (1, 1), 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (5, 5), (1, 1), 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, input):
        output = self.model(input)
        return output
phd = Phd()
phd = phd.cuda()


# 损失函数
loss_fun = nn.CrossEntropyLoss()
loss_fun = loss_fun.cuda()

# 优化器
learning_rate = 1e-2
print("学习率：{}".format(learning_rate))
optimizer = torch.optim.SGD(phd.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
total_train_step = 0  # 训练次数
total_test_step = 0   # 测试次数
epoch = 1            # 训练轮数

# 添加tensorboard
writer = SummaryWriter("/logs_train")

# 开始时间
start_time = time.time()


for i in range(epoch):
    print("----------第{}轮训练开始----------".format(i+1))

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = phd(imgs)
        loss = loss_fun(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step+1
        if (total_train_step+1) % 100 == 0:
            end_time = time.time()
            print("time:{:.2f},训练次数：{}，loss：{:3f}".format(end_time-start_time,total_train_step+1, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = phd(imgs)
            loss = loss_fun(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy +accuracy

    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_loss + 1

    # 保存文件
    if i + 1 == 1:
       torch.save(phd, "phd_{}_gpu.pth".format(i+1))
       print("模型已保存")

writer.close()



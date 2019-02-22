# encoding:utf-8
import os
import common
import random
import numpy as np
from PIL import Image
from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as T
from torchvision.transforms import functional
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from torchvision import models
from torch.utils import data
import cv2


# 图片加载类
class MyDataset(data.Dataset):
    def __init__(self, root_dir, label_file, img_size, transforms=None, is_train=False):
        self.root_dir = root_dir
        records_txt = common.read_data(label_file, 'r')
        self.records = records_txt.split('\n')
        self.img_size = img_size
        self.is_train = is_train

        # imgs = os.listdir(root)
        # self.imgs = [os.path.join(root, img) for img in imgs]
        # self.label_path = label_path
        self.transforms = transforms

    def __getitem__(self, index):
        record = self.records[index]
        str_list = record.split(" ")
        img_file = os.path.join(self.root_dir, str_list[0])

        img = Image.open(img_file)
        old_size = img.size[0]

        label = str_list[2:]
        label = map(float, label)
        label = np.array(label)

        # if self.is_train:                                               # 训练模式，才做变换
        #     img, label = self.RandomHorizontalFlip(img, label)          # 图片做随机水平翻转
        #     self.show_img(img, label)

        label = label * self.img_size / old_size
        if self.transforms:
            img = self.transforms(img)

        return img, label, img_file

    def __len__(self):
        return len(self.records)


class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN, self).__init__()

        self.batch_1 = nn.BatchNorm2d(3)
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.batch_2 = nn.BatchNorm2d(32)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.batch_3 = nn.BatchNorm2d(32)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        self.dropout_1 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(in_features=64*13*2, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x1 = self.batch_1(x)
        x1 = self.conv_1(x1)
        # print('conv1', x1.size())
        x1 = F.relu(x1)
        x1 = F.max_pool2d(x1, kernel_size=2, stride=2, padding=0)
        # print('max_pool2d', x1.size())

        x2 = self.batch_2(x1)
        x2 = self.conv_2(x2)
        # print('conv2', x2.size())
        x2 = F.relu(x2)
        x2 = F.max_pool2d(x2, kernel_size=2, stride=2, padding=0)
        # print('max_pool2d', x2.size())

        x3 = self.batch_3(x2)
        x3 = self.conv_3(x3)
        # print('conv3', x3.size())
        x3 = F.relu(x3)
        x3 = F.max_pool2d(x3, kernel_size=2, stride=2, padding=0)
        # print('max_pool2d', x3.size())

        x4 = self.dropout_1(x3)
        x4 = x4.view(x4.size()[0], -1)
        # print('view', x4.size())
        x4 = F.relu(self.fc1(x4))
        # print('fc1', x4.size())
        x4 = F.relu(self.fc2(x4))
        # print('fc2', x4.size())
        x4 = F.softmax(x4, dim=1)
        # print('softmax', x4.size())
        output = x4
        return output

    def load(self, name):
        print('[Load model] %s...' % name)
        self.load_state_dict(torch.load(name))

    def save(self, name):
        print('[Save model] %s ...' % name)
        torch.save(self.state_dict(), name)


class ModuleTrain():
    def __init__(self, train_path, test_path, model_file, model=CNN(), num_classes=4, img_size=(118, 30), batch_size=8, lr=1e-3,
                 re_train=False, best_acc=0.9):
        self.train_path = train_path
        self.test_path = test_path
        self.model_file = model_file
        self.img_size = img_size
        self.batch_size = batch_size
        self.re_train = re_train                        # 不加载训练模型，重新进行训练
        self.best_acc = best_acc                        # 最好的损失值，小于这个值，才会保存模型
        self.num_classes = num_classes

        if torch.cuda.is_available():
            self.use_gpu = True
        else:
            self.use_gpu = False

        print('[ModuleCNN]')
        print('train_path: %s' % self.train_path)
        print('test_path: %s' % self.test_path)
        print('img_size: ' + str(self.img_size))
        print('batch_size: %d' % self.batch_size)

        # 模型
        self.model = model

        if self.use_gpu:
            print('[use gpu] ...')
            self.model = self.model.cuda()

        # 加载模型
        if os.path.exists(self.model_file) and not self.re_train:
            self.load(self.model_file)

        # RandomHorizontalFlip
        self.transform_train = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
        ])

        self.transform_test = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

        # # Dataset
        # train_label = os.path.join(self.train_path, 'label.txt')
        # train_dataset = MyDataset(self.train_path, train_label, self.img_size, self.transform_test, is_train=True)
        # test_label = os.path.join(self.test_path, 'label.txt')
        # test_dataset = MyDataset(self.test_path, test_label, self.img_size, self.transform_test, is_train=False)
        train_dataset = ImageFolder(self.train_path, transform=self.transform_train)
        test_dataset = ImageFolder(self.test_path, transform=self.transform_test)
        # print(train_dataset.class_to_idx)

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        # self.loss = F.mse_loss
        # self.loss = F.smooth_l1_loss
        # self.loss = F.cross_entropy
        self.loss = torch.nn.CrossEntropyLoss()
        # self.loss = F.nll_loss

        self.lr = lr
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.5)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        pass

    def train(self, epoch, decay_epoch=40, save_best=True):
        print('[train] epoch: %d' % epoch)
        for epoch_i in range(epoch):

            if epoch_i >= decay_epoch and epoch_i % decay_epoch == 0:                   # 减小学习速率
                self.lr = self.lr * 0.1
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

            print('================================================')
            self.model.train()
            for batch_idx, (data, label) in enumerate(self.train_loader):
                # label = label.view(-1, 1)
                # print('label:', label)
                # label = torch.zeros(self.batch_size, self.num_classes, dtype=torch.long).scatter_(1, label, 1.)
                # print('label:', label)

                data, label = Variable(data), Variable(label)

                if self.use_gpu:
                    data = data.cuda()
                    label = label.cuda()

                # 梯度清0
                self.optimizer.zero_grad()

                # 计算损失
                output = self.model(data)

                loss = self.loss(output, label)
                # if self.use_gpu:
                #     loss = self.loss(output.type(torch.cuda.LongTensor), label.type(torch.cuda.LongTensor))
                # else:
                #     loss = self.loss(output.type(torch.LongTensor), label.type(torch.LongTensor))

                # 反向传播计算梯度
                loss.backward()

                # 更新参数
                self.optimizer.step()

                # update
                if batch_idx == 0:
                    print('[Train] Epoch: {} [{}/{}]\tLoss: {:.6f}\tlr: {}'.format(epoch_i, batch_idx * len(data),
                        len(self.train_loader.dataset), loss.item()/self.batch_size, self.lr))

            test_acc = self.test()
            if save_best is True:
                if self.best_acc < test_acc:
                    self.best_acc = test_acc
                    str_list = self.model_file.split('.')
                    best_model_file = ""
                    for str_index in range(len(str_list)):
                        best_model_file = best_model_file + str_list[str_index]
                        if str_index == (len(str_list) - 2):
                            best_model_file += '_best'
                        if str_index != (len(str_list) - 1):
                            best_model_file += '.'
                    self.save(best_model_file)                                  # 保存最好的模型

        self.save(self.model_file)

    def test(self, show_info=False):
        total_count = 0.
        success_count = 0.

        # 测试集
        self.model.eval()
        for data, target in self.test_loader:
            data, target = Variable(data), Variable(target)

            if self.use_gpu:
                data = data.cuda()
                target = target.cuda()

            output = self.model(data)
            label = torch.max(input=output, dim=1)[1]
            # sum up batch loss
            # loss = self.loss(output, target)
            # test_loss += loss.item()

            if show_info is True:
                print('true_target', target)
                print('  pre_label', label)

            for y, y_ in zip(target, label):
                total_count += 1.
                if y.item() == y_.item():
                    success_count += 1.

            # if show_img:
            #     for i in range(len(output[:, 1])):
            #         self.show_img(img_files[i], output[i].cpu().detach().numpy(), target[i].cpu().detach().numpy())

        acc = success_count / total_count
        print('[Test] set: Acc: {:.4f}\n'.format(acc))
        return acc

    def load(self, name):
        print('[Load model] %s ...' % name)
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        print('[Save model] %s ...' % name)
        torch.save(self.model.state_dict(), name)


if __name__ == '__main__':
    # loss = nn.CrossEntropyLoss()
    # input = torch.randn(3, 5, requires_grad=True)
    # print(input)
    # target = torch.empty(3, dtype=torch.long).random_(5)
    # print(target)
    # output = loss(input, target)
    # print(output)
    # output.backward()
    # print(output)

    # x = torch.rand(1, 5)
    # label = torch.zeros(3, 5).scatter_(0, torch.LongTensor([[2, 0, 0, 1, 2]]), x)
    # print(x)
    # print(label)
    # z = torch.zeros(2, 4).scatter_(1, torch.LongTensor([[2], [3]]), 1.23)
    # print(z)

    # model = CNN(class_num=4)
    # data = Variable(torch.randn(10, 3, 118, 30))
    # x = model(data)
    # print('x', x.size())

    # ====================================================================================================
    train_path = './Data/train'
    test_path = './Data/train'

    FILE_PATH = './Model/cnn_params_100.pkl'
    # FILE_PATH = './Model/cnn_params_best.pkl'
    # FILE_PATH = './Model/cnn_params_100.pkl'

    model = CNN(num_classes=3)
    model_train = ModuleTrain(train_path, test_path, FILE_PATH, num_classes=3, img_size=(118, 30), lr=1e-3)

    # FILE_PATH = './Model/resnet18_params_99.10.pkl'
    # model = models.resnet18(num_classes=3)
    # model_train = ModuleTrain(train_path, test_path, FILE_PATH, model=model, batch_size=8, img_size=(224, 224), lr=1e-3)

    model_train.train(200, 60)
    # model_train.test(show_info=True)









"""
使用torchvision的datasets和transforms模块,加载CIFAR10数据集,并使用CNN模型进行训练和测试
加载数据集
搭建模型
训练模型
测试模型

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

def create_dataset():
    #获取训练集
    train_dataset=datasets.CIFAR10(root='./data',train=True,download=True,transform=ToTensor())
    #获取测试集
    test_dataset=datasets.CIFAR10(root='./data',train=False,download=True,transform=ToTensor())
    #返回训练集和测试集
    return train_dataset,test_dataset


if __name__ == "__main__":
    train_dataset,test_dataset=create_dataset()
    print(f"train_dataset: {train_dataset},shape: {train_dataset.shape}")
    print(f"test_dataset: {test_dataset},shape: {test_dataset.shape}")
    print(f'class_names: {train_dataset.class_to_idx}')
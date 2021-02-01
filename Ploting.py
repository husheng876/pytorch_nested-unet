#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :Ploting.py
# @Time      :2020/10/22 18:17
# @Author    :HuSheng
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import time
import os
from PIL import Image

#本文件中的函数用来绘制loss,,iou,val_loss,val_iou的图像，读取的文件是.csv的文件

def get_loss_graph(x, y1,y2):
    plt.title('loss_result')
    plt.plot(x, y1, color='red', marker='|', label='train_loss')

    plt.plot(x, y2, color='blue', marker='|', label='val_loss')

    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def get_iou_graph(x,y1,y2):
    plt.plot(x, y1, color='red', marker='|', label='train_iou')
    plt.plot(x, y2, color='blue', marker='|', label='val_iou')
    plt.xlabel('epoch')
    plt.ylabel('iou')
    plt.legend()
    plt.show()

def plot_data(path,x="epoch",loss="loss",iou="iou",val_loss="val_loss",val_iou="val_iou"):#参数12分别是图像的横纵坐标用于实现
    epochs = []
    losses = []
    ious = []
    val_losses = []
    val_ious=[]
    couter_num = 1
    csv_path=path
    with open(csv_path, 'r') as csvfile:
        pre_reader = csv.DictReader(csvfile)  # 直接生成一个pre_reader，用于迭代读出

        for reader in pre_reader:  # 迭代读出pre_reader里的数据
            couter_num+=1
            epoch = reader[x]
            lossx = reader[loss]  # 取出其中loss
            ioux=reader[iou]
            val_lossx=reader[val_loss]
            val_ioux=reader[val_iou]
            if couter_num%10==0:#每10个epoch输出一次
                epochs.append(int(epoch))
                losses.append(float(lossx))
                ious.append(float(ioux))
                val_ious.append(float(val_ioux))
                val_losses.append(float(val_lossx))
    get_loss_graph(epochs, losses,val_losses)
    get_iou_graph(epochs, ious, val_ious)
if __name__ == "__main__":

    plot_data(path=r'D:\results\ISIC\vanilla\log.csv')



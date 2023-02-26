import torch
import torchvision
from core_utils import main
from core_utils.utils import data_handling, train, test, gradcam, helpers, augmentation
from core_utils.models import resnet, s8_custom_resnet
from pprint import pprint
# from torch_lr_finder import LRFinder


import timm
import urllib
import torch
import os
import numpy as np

from torch import nn
import torchvision.transforms as T
import torch.nn.functional as F

from PIL import Image
from S9.datamodule import CIFARDataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from matplotlib.colors import LinearSegmentedColormap
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from torchmetrics import Accuracy


# n_out = (n_in - k + 2*p)/s + 1
# j_out = j_in * s
# r_out = r_in + (k - 1) * j_in
# j_in = j_out_previous, initially 1

# Output size = (Input size + 2 * padding - dilation * (kernel size - 1) - 1) / stride + 1

# n_out = (n_in + 2* p - d*(k-1) - 1)/s + 1

# so if d == 1 then n_out = (n_in + 2*p - k)/s + 1
# so if d == 2 then n_out = (n_in + 2*p - 2k + 1)/s + 1
# so if d == 3 then n_out = (n_in + 2*p - 3k + 2)/s + 1

import torch.nn as nn

class PrintShape(nn.Module):
    def __init__(self, text=None):
        super(PrintShape, self).__init__()
        self.text = text

    def forward(self, x):
        print("For the: ", self.text, " the shape is: ", x.shape)
        return x



class ULTIMUS(nn.Module):
    def __init__(self):
        super(ULTIMUS, self).__init__()
        self.key  = nn.Linear(in_features=48, out_features=8, bias=False)
        self.query = nn.Linear(in_features=48, out_features=8, bias=False)
        self.value = nn.Linear(in_features=48, out_features=8, bias=False)

        self.softmax = nn.Softmax(dim=1)

        self.out = nn.Linear(in_features=8, out_features=48, bias=False)


    def forward(self, x):

       
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        softmax_output = self.softmax(torch.matmul(query.T, key)/torch.sqrt(torch.tensor(8.0)))

        pre_output = torch.matmul(value, softmax_output)

        output = self.out(pre_output)

        return output


class Att_Model(torch.nn.Module):
    def __init__(self):
        super(Att_Model, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Here n_in=32, n_out=32, k = 3, p = 1, s = 1, d = 1, j_in=1, r_out = 3

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Here n_in=32, n_out=32, k = 3, p = 1, s = 1, d = 1, j_in=1, r_out = 5

            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
            # Here n_in=32, n_out=32, k = 3, p = 1, s = 1, d = 1, j_in=1, r_out = 7
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.ultimus_1 = ULTIMUS()
        self.ultimus_2 = ULTIMUS()
        self.ultimus_3 = ULTIMUS()
        self.ultimus_4 = ULTIMUS()

        

        self.final_layer = nn.Linear(in_features=48, out_features=10, bias=False)

        
    def forward(self, x):

        x = self.conv_layer(x)

        average_pooled = self.avg_pool(x)

        # squash the output of the average pool layer
        average_pooled = average_pooled.view(average_pooled.size(0), -1)
        
        ultimus_1_output = self.ultimus_1(average_pooled)

        ultimus_2_output = self.ultimus_2(ultimus_1_output)

        ultimus_3_output = self.ultimus_3(ultimus_2_output)

        ultimus_4_output = self.ultimus_4(ultimus_3_output)

        final_output = self.final_layer(ultimus_4_output)

        return final_output



        


# print model summary using torchsummary
# eva = Att_Model()
# eva = eva.to("cuda")
# from torchsummary import summary
# summary(eva, input_size=(3, 32, 32))
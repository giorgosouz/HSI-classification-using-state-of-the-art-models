 # -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init
# utils
import math
import os
import datetime
import numpy as np
from sklearn.externals import joblib
from tqdm import tqdm
from utils import grouper, sliding_window, count_sliding_window,\
                  camel_to_snake
from custom_activation import PRTanh
import time

def get_model(name, **kwargs):
    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        model: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    device = kwargs.setdefault('device', torch.device('cpu'))
    n_classes = kwargs['n_classes']
    n_bands = kwargs['n_bands']
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
    weights = weights.to(device)
    weights = kwargs.setdefault('weights', weights)

    if name == 'nn':
        kwargs.setdefault('patch_size', 1)
        center_pixel = True
        model = Baseline(n_bands, n_classes,
                         kwargs.setdefault('dropout', False))
        lr = kwargs.setdefault('learning_rate', 0.1)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('epoch', 100)
        kwargs.setdefault('batch_size', 4)
    elif name == 'hamida':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        model = HamidaEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('learning_rate', 0.00001)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        kwargs.setdefault('batch_size', 4)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('epoch', 100)
    elif name == 'lee':
        kwargs.setdefault('epoch', 300)
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        model = LeeEtAl(n_bands, n_classes)
        lr = kwargs.setdefault('learning_rate', 0.0001)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        kwargs.setdefault('batch_size', 8)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'chen':
        patch_size = kwargs.setdefault('patch_size', 27)
        center_pixel = True
        model = ChenEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('learning_rate', 0.1)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('epoch', 10)
        kwargs.setdefault('batch_size', 230)
    elif name == 'li':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        model = LiEtAl(n_bands, n_classes, n_planes=16, patch_size=patch_size)
        lr = kwargs.setdefault('learning_rate', 0.0001)
        optimizer = optim.SGD(model.parameters(),
                              lr=lr, momentum=0.9, weight_decay=0.0005)
        epoch = kwargs.setdefault('epoch', 200)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('scheduler', optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch // 2, (5 * epoch) // 6], gamma=0.1))
        kwargs.setdefault('batch_size', 4)
    elif name == 'hu':
        kwargs.setdefault('patch_size', 1)
        center_pixel = True
        model = HuEtAl(n_bands, n_classes)
        # From what I infer from the paper (Eq.7 and Algorithm 1), it is standard SGD with lr = 0.01
        lr = kwargs.setdefault('learning_rate', 0.1)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('epoch', 100)
        kwargs.setdefault('batch_size', 4)
    elif name == 'he':
        # We train our model by AdaGrad [18] algorithm, in which
        # the base learning rate is 0.01. In addition, we set the batch
        # as 40, weight decay as 0.01 for all the layers
        # The input of our network is the HSI 3D patch in the size of 7×7×Band
        kwargs.setdefault('patch_size', 7)
        kwargs.setdefault('batch_size', 4)
        lr = kwargs.setdefault('learning_rate', 0.001)
        center_pixel = True
        model = HeEtAl(n_bands, n_classes, patch_size=kwargs['patch_size'])
        # For Adagrad, we need to load the model on GPU before creating the optimizer
        model = model.to(device)
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'luo':
        # All  the  experiments  are  settled  by  the  learning  rate  of  0.1,
        # the  decay  term  of  0.09  and  batch  size  of  100.
        kwargs.setdefault('patch_size', 3)
        kwargs.setdefault('batch_size', 4)
        lr = kwargs.setdefault('learning_rate', 0.0001 )
        center_pixel = True
        model = LuoEtAl(n_bands, n_classes, patch_size=kwargs['patch_size'])
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'sharma':
        # We train our S-CNN from scratch using stochastic gradient descent with
        # momentum set to 0.9, weight decay of 0.0005, and with a batch size
        # of 60.  We initialize an equal learning rate for all trainable layers
        # to 0.05, which is manually decreased by a factor of 10 when the validation
        # error stopped decreasing. Prior to the termination the learning rate was
        # reduced two times at 15th and 25th epoch. [...]
        # We trained the network for 30 epochs
        kwargs.setdefault('batch_size', 60)
        epoch = kwargs.setdefault('epoch', 30)
        lr = kwargs.setdefault('lr', 0.05)
        center_pixel = True
        # We assume patch_size = 64
        kwargs.setdefault('patch_size', 64)
        model = SharmaEtAl(n_bands, n_classes, patch_size=kwargs['patch_size'])
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('scheduler', optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch // 2, (5 * epoch) // 6], gamma=0.1))
    elif name == 'liu':
        kwargs['supervision'] = 'full'
        # "The learning rate is set to 0.001 empirically. The number of epochs is set to be 40."
        kwargs.setdefault('epoch', 80)
        lr = kwargs.setdefault('lr', 0.0001)
        center_pixel = True 
        patch_size = kwargs.setdefault('patch_size', 9)
        model = LiuEtAl(n_bands, n_classes, patch_size)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        # "The unsupervised cost is the squared error of the difference"
        criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']),F.mse_loss)
        kwargs.setdefault('batch_size', 8)
    elif name == 'boulch':
        kwargs['supervision'] = 'semi'
        kwargs.setdefault('patch_size', 1)
        kwargs.setdefault('epoch', 100)
        lr = kwargs.setdefault('lr', 0.1)
        center_pixel = True
        model = BoulchEtAl(n_bands, n_classes)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']), lambda rec, data: F.mse_loss(rec, data.squeeze()))
    elif name == 'mou':
        kwargs.setdefault('patch_size', 1)
        center_pixel = True
        kwargs.setdefault('epoch', 50)
        # "The RNN was trained with the Adadelta algorithm [...] We made use of a
        # fairly  high  learning  rate  of  1.0  instead  of  the  relatively  low
        # default of  0.002 to  train the  network"
        lr = kwargs.setdefault('lr', 0.01)
        model = MouEtAl(n_bands, n_classes)
        kwargs.setdefault('batch_size', 64)
        # For Adadelta, we need to load the model on GPU before creating the optimizer
        model = model.to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    else:
        raise KeyError("{} model is unknown.".format(name))

    model = model.to(device)
    epoch = kwargs.setdefault('epoch', 100)
    kwargs.setdefault('scheduler', optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True))
    #kwargs.setdefault('scheduler', None)
    kwargs.setdefault('batch_size', 128)
    kwargs.setdefault('supervision', 'full')
    kwargs.setdefault('flip_augmentation', False)
    kwargs.setdefault('radiation_augmentation', False)
    kwargs.setdefault('mixture_augmentation', False)
    kwargs['center_pixel'] = center_pixel
    return model, optimizer, criterion, kwargs


class Baseline(nn.Module):
    """
    Baseline network
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, dropout=False):
        super(Baseline, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(input_channels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, n_classes)

        self.apply(self.weight_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc4(x)
        return x

class HuEtAl(nn.Module):
    """
    Deep Convolutional Neural Networks for Hyperspectral Image Classification
    Wei Hu, Yangyu Huang, Li Wei, Fan Zhang and Hengchao Li
    Journal of Sensors, Volume 2015 (2015)
    https://www.hindawi.com/journals/js/2015/258619/
    """
    @staticmethod
    def weight_init(m):
        # [All the trainable parameters in our CNN should be initialized to
        # be a random value between −0.05 and 0.05.]
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.pool(self.conv(x))
        return x.numel() 

    def __init__(self, input_channels, n_classes, kernel_size=None, pool_size=None):
        super(HuEtAl, self).__init__()
        if kernel_size is None:
           # [In our experiments, k1 is better to be [ceil](n1/9)]
           kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
           # The authors recommand that k2's value is chosen so that the pooled features have 30~40 values
           # ceil(kernel_size/5) gives the same values as in the paper so let's assume it's okay
           pool_size = math.ceil(kernel_size / 5)
        self.input_channels = input_channels

        # [The first hidden convolution layer C1 filters the n1 x 1 input data with 20 kernels of size k1 x 1]
        self.conv = nn.Conv1d(1, 20, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()
        # [n4 is set to be 100]
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)
        self.apply(self.weight_init)

    def forward(self, x):
        # [In our design architecture, we choose the hyperbolic tangent function tanh(u)]
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = F.relu(self.pool(x))
        x = x.view(-1, self.features_size)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class HamidaEtAl(nn.Module):
    """
    3-D Deep Learning Approach for Remote Sensing Image Classification
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    IEEE TGRS, 2018
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=5, dilation=1):
        super(HamidaEtAl, self).__init__()
        # The first layer is a (3,3,3) kernel sized Conv characterized
        # by a stride equal to 1 and number of neurons equal to 20
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = (dilation, 1, 1)

        if patch_size == 3:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=(1,0,0))
        else:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=(1,0,0))
        # Next pooling is applied using a layer identical to the previous one
        # with the difference of a 1D kernel size (1,1,3) and a larger stride
        # equal to 2 in order to reduce the spectral dimension
        self.pool1 = nn.Conv3d(
            20, 2, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        # Then, a duplicate of the first and second layers is created with
        # 35 hidden neurons per layer.
        self.conv2 = nn.Conv3d(
            2, 35, (3, 3, 3), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
        self.pool2 = nn.Conv3d(
            35, 2, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        # Finally, the 1D spatial dimension is progressively reduced
        # thanks to the use of two Conv layers, 35 neurons each,
        # with respective kernel sizes of (1,1,3) and (1,1,2) and strides
        # respectively equal to (1,1,1) and (1,1,2)
        self.conv3 = nn.Conv3d(
            2, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
        self.pool3 = nn.Conv3d(
            35, 2, (1, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        self.conv4 = nn.Conv3d(
            2, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
        self.pool4 = nn.Conv3d(
            35, 4, (1, 1, 1), dilation=dilation, stride=(2, 2, 2), padding=(0, 0, 0))

        
        self.dropout = nn.Dropout(p=0.5)

        self.features_size = self._get_final_flattened_size()
        # The architecture ends with a fully connected layer where the number
        # of neurons is equal to the number of input classes.
        self.fc = nn.Linear(self.features_size, n_classes)
        

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.pool3(self.conv3(x))
            x = self.pool4(self.conv4(x))
            _, t, c, w, h = x.size()
        return t * c * w * h

#    def forward(self, x):
#        x = F.relu(self.conv1(x))
#        x = F.relu(self.pool1(x))
#        x = F.relu(self.conv2(x))
#        x = F.relu(self.pool2(x))
#        x = F.relu(self.conv3(x))
#        x = F.relu(self.pool3(x))
#        x = F.relu(self.conv4(x))
#        x = F.relu(self.pool4(x))
#        x = x.view(-1, self.features_size)
#        x = self.dropout(x)
#        x = self.fc(x)
#        return x
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = x.view(-1, self.features_size)
        x = self.dropout(x)
        x = self.fc(x)
        return x



class LeeEtAl(nn.Module):
    """
    CONTEXTUAL DEEP CNN BASED HYPERSPECTRAL CLASSIFICATION
    Hyungtae Lee and Heesung Kwon
    IGARSS 2016
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, in_channels, n_classes):
        super(LeeEtAl, self).__init__()
        # The first convolutional layer applied to the input hyperspectral
        # image uses an inception module that locally convolves the input
        # image with two convolutional filters with different sizes
        # (1x1xB and 3x3xB where B is the number of spectral bands)
        self.conv_3x3 = nn.Conv3d(
            1, 128, (in_channels, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv_1x1 = nn.Conv3d(
            1, 128, (in_channels, 1, 1), stride=(1, 1, 1), padding=(0,0,0))
        self.conv_5x5 = nn.Conv3d(
            1, 128, (in_channels, 5, 5), stride=( 1,1, 1), padding=(0, 0, 0))    

        # We use two modules from the residual learning approach
        # Residual block 1
        self.conv1 = nn.Conv2d(384, 128, (1, 1))
        self.conv2 = nn.Conv2d(128, 128, (1, 1))
        self.conv3 = nn.Conv2d(128, 128, (1, 1))

        # Residual block 2
        self.conv4 = nn.Conv2d(128, 128, (1, 1))
        self.conv5 = nn.Conv2d(128, 128, (1, 1))

        # The layer combination in the last three convolutional layers
        # is the same as the fully connected layers of Alexnet
        self.conv6 = nn.Conv2d(128, 128, (1, 1))
        self.conv7 = nn.Conv2d(128, 128, (1, 1))
        self.conv8 = nn.Conv2d(128, n_classes, (1, 1))

        self.lrn1 = nn.LocalResponseNorm(256)
        self.lrn2 = nn.LocalResponseNorm(128)
        self.pool1 = nn.MaxPool3d((1,5,5), stride=1)
        self.pool2 = nn.MaxPool3d((1,3,3), stride=1)

        # The 7 th and 8 th convolutional layers have dropout in training
        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def forward(self, x):
        # Inception module
        x_1x1 = self.pool1(torch.squeeze(self.conv_1x1(x)))
        x_3x3 = self.pool2(torch.squeeze(self.conv_3x3(x)))
        x_5x5 = torch.squeeze(self.conv_5x5(x),2)
        
        
        x = torch.cat([x_5x5,x_3x3, x_1x1], dim=1)
        # Remove the third dimension of the tensor
#        x = torch.squeeze(x)

        # Local Response Normalization
        x = F.relu(self.lrn1(x))
        
#
#        # First convolution
        x = self.conv1(x)
        
##
##        # Local Response Normalization
        x = F.relu(self.lrn2(x))
#
#        # First residual block
        x_res = F.relu(self.conv2(x))
        x_res = self.conv3(x_res)
        x = F.relu(x + x_res)
#
#        # Second residual block
        x_res = F.relu(self.conv4(x))
        x_res = self.conv5(x_res)
        x = F.relu(x + x_res)
#
        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        x = self.dropout(x)
        x = self.conv8(x)
        x = torch.squeeze(x)
        return x
        
        
       


class ChenEtAl(nn.Module):
    """
    DEEP FEATURE EXTRACTION AND CLASSIFICATION OF HYPERSPECTRAL IMAGES BASED ON
                        CONVOLUTIONAL NEURAL NETWORKS
    Yushi Chen, Hanlu Jiang, Chunyang Li, Xiuping Jia and Pedram Ghamisi
    IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2017
    """
    @staticmethod
    def weight_init(m):
        # In the beginning, the weights are randomly initialized
        # with standard deviation 0.001
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.normal_(m.weight, std=0.001)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=27):
        super(ChenEtAl, self).__init__()
        self.input_channels = input_channels
        
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1,32, (32, 4, 4))
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, (32, 5, 5))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.conv3 = nn.Conv3d(64, 128, (32, 4, 4))


        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        #x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = x.view(-1, self.features_size)
        x = self.fc(x)
        return x


class LiEtAl(nn.Module):
    """
    SPECTRAL–SPATIAL CLASSIFICATION OF HYPERSPECTRAL IMAGERY
            WITH 3D CONVOLUTIONAL NEURAL NETWORK
    Ying Li, Haokui Zhang and Qiang Shen
    MDPI Remote Sensing, 2017
    http://www.mdpi.com/2072-4292/9/1/67
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data, 0)

    def __init__(self, input_channels, n_classes, n_planes=2, patch_size=5):
        super(LiEtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        # The proposed 3D-CNN model has two 3D convolution layers (C1 and C2)
        # and a fully-connected layer (F1)
        # we fix the spatial size of the 3D convolution kernels to 3 × 3
        # while only slightly varying the spectral depth of the kernels
        # for the Pavia University and Indian Pines scenes, those in C1 and C2
        # were set to seven and three, respectively
        self.conv1 = nn.Conv3d(1, n_planes, (7, 3, 3), padding=(1, 0, 0))
        # the number of kernels in the second convolution layer is set to be
        # twice as many as that in the first convolution layer
        self.conv2 = nn.Conv3d(n_planes, 2 * n_planes,
                               (3, 3, 3), padding=(1, 0, 0))
        #self.dropout = nn.Dropout(p=0.5)
        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.conv1(x)
            x = self.conv2(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        #x = self.dropout(x)
        x = self.fc(x)
        return x

class HeEtAl(nn.Module):
    """
    MULTI-SCALE 3D DEEP CONVOLUTIONAL NEURAL NETWORK FOR HYPERSPECTRAL
    IMAGE CLASSIFICATION
    Mingyi He, Bo Li, Huahui Chen
    IEEE International Conference on Image Processing (ICIP) 2017
    https://ieeexplore.ieee.org/document/8297014/
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=7):
        super(HeEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, 16, (11, 3, 3), stride=(3,1,1))
        self.conv2_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0,0,0))
        self.conv2_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1,0,0))
        self.conv2_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2,0,0))
        self.conv2_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5,0,0))
        self.conv3_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0,0,0))
        self.conv3_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1,0,0))
        self.conv3_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2,0,0))
        self.conv3_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5,0,0))
        self.conv4 = nn.Conv3d(16, 16, (3, 2, 2))
        self.pooling = nn.MaxPool2d((3,2,2), stride=(3,2,2))
        # the ratio of dropout is 0.6 in our experiments
        self.dropout = nn.Dropout(p=0.6)

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.conv1(x)
            x2_1 = self.conv2_1(x)
            x2_2 = self.conv2_2(x)
            x2_3 = self.conv2_3(x)
            x2_4 = self.conv2_4(x)
            x = x2_1 + x2_2 + x2_3 + x2_4
            x3_1 = self.conv3_1(x)
            x3_2 = self.conv3_2(x)
            x3_3 = self.conv3_3(x)
            x3_4 = self.conv3_4(x)
            x = x3_1 + x3_2 + x3_3 + x3_4
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        x2_4 = self.conv2_4(x)
        x = x2_1 + x2_2 + x2_3 + x2_4
        x = F.relu(x)
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x)
        x3_3 = self.conv3_3(x)
        x3_4 = self.conv3_4(x)
        x = x3_1 + x3_2 + x3_3 + x3_4
        x = F.relu(x)
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class LuoEtAl(nn.Module):
    """
    HSI-CNN: A Novel Convolution Neural Network for Hyperspectral Image
    Yanan Luo, Jie Zou, Chengfei Yao, Tao Li, Gang Bai
    International Conference on Pattern Recognition 2018
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=3, n_planes=90):
        super(LuoEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.n_planes = n_planes

        # the 8-neighbor pixels [...] are fed into the Conv1 convolved by n1 kernels
        # and s1 stride. Conv1 results are feature vectors each with height of and
        # the width is 1. After reshape layer, the feature vectors becomes an image-like 
        # 2-dimension data.
        # Conv2 has 64 kernels size of 3x3, with stride s2.
        # After that, the 64 results are drawn into a vector as the input of the fully 
        # connected layer FC1 which has n4 nodes.
        # In the four datasets, the kernel height nk1 is 24 and stride s1, s2 is 9 and 1
        self.conv1 = nn.Conv3d(1, n_planes, (24, 3, 3), padding=0, stride=(9,1,1))
        self.conv2 = nn.Conv2d(1, 64, (3, 3), stride=(1, 1))



# estw oti pool einai 3x3 giati leei xekathara gia pool 
        self.pool = nn.MaxPool2d((2,2))

        
        self.features_size = self._get_final_flattened_size()
        
       

        self.fc1 = nn.Linear(self.features_size, 1024)
        self.fc2 = nn.Linear(1024, 100)
        self.fc3 = nn.Linear(100, n_classes)
        

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.conv1(x)
            b = x.size(0)
            x = x.view(b, 1, -1, self.n_planes)
            x = self.conv2(x)
            x = self.pool(x)
            _, c, w, h = x.size()
        return c * w * h

    def forward(self, x):
        x = (self.conv1(x))
        b = x.size(0)     
        x = x.view(b, 1, -1, self.n_planes)
        
        x = (self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, self.features_size)
      #  print (x.shape)
        x = (self.fc1(x))
        x = (self.fc2(x))
        x = (self.fc3(x))
       
        return x


class SharmaEtAl(nn.Module):
    """
    HYPERSPECTRAL CNN FOR IMAGE CLASSIFICATION & BAND SELECTION, WITH APPLICATION
    TO FACE RECOGNITION
    Vivek Sharma, Ali Diba, Tinne Tuytelaars, Luc Van Gool
    Technical Report, KU Leuven/ETH Zürich
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=64):
        super(SharmaEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        # An input image of size 263x263 pixels is fed to conv1
        # with 96 kernels of size 6x6x96 with a stride of 2 pixels
        self.conv1 = nn.Conv3d(1, 96, (input_channels, 6, 6), stride=(1,2,2))
        self.conv1_bn = nn.BatchNorm3d(96)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        #  256 kernels of size 3x3x256 with a stride of 2 pixels
        self.conv2 = nn.Conv3d(1, 256, (96, 3, 3), stride=(1,2,2))
        self.conv2_bn = nn.BatchNorm3d(256)
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        # 512 kernels of size 3x3x512 with a stride of 1 pixel
        self.conv3 = nn.Conv3d(1, 512, (256, 3, 3), stride=(1,1,1))
        # Considering those large kernel values, I assume they actually merge the
        # 3D tensors at each step

        self.features_size = self._get_final_flattened_size()

        # The fc1 has 1024 outputs, where dropout was applied after
        # fc1 with a rate of 0.5
        self.fc1 = nn.Linear(self.features_size, 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = F.relu(self.conv1_bn(self.conv1(x)))
            x = self.pool1(x)
            print(x.size())
            b, t, c, w, h = x.size()
            x = x.view(b, 1, t*c, w, h) 
            x = F.relu(self.conv2_bn(self.conv2(x)))
            x = self.pool2(x)
            print(x.size())
            b, t, c, w, h = x.size()
            x = x.view(b, 1, t*c, w, h) 
            x = F.relu(self.conv3(x))
            print(x.size())
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.pool1(x)
        b, t, c, w, h = x.size()
        x = x.view(b, 1, t*c, w, h) 
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool2(x)
        b, t, c, w, h = x.size()
        x = x.view(b, 1, t*c, w, h) 
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.features_size)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x



class LiuEtAl(nn.Module):
    """
    A semi-supervised convolutional neural network for hyperspectral image classification
    Bing Liu, Xuchu Yu, Pengqiang Zhang, Xiong Tan, Anzhu Yu, Zhixiang Xue
    Remote Sensing Letters, 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=9):
        super(LiuEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.aux_loss_weight = 1
        self.s2 = 0.001
        self.r = torch.randn
        self.loss = nn.Tanh()
        self.n_classes = n_classes
        
        
        
        # "W1 is a 3x3xB1 kernel [...] B1 is the number of the output bands for the convolutional
        # "and pooling layer" -> actually 3x3 2D convolutions with B1 outputs
        # "the value of B1 is set to be 80"
        self.conv1 = nn.Conv2d(input_channels, 80, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv1_bn = nn.BatchNorm2d(80)

        self.features_sizes = self._get_sizes()

        self.fc_enc = nn.Linear(self.features_sizes[2], n_classes)

        # Decoder
        self.fc1_dec = nn.Linear(n_classes, self.features_sizes[2])
        self.fc1_dec_bn = nn.BatchNorm1d(self.features_sizes[2])
        self.fc2_dec = nn.Linear(self.features_sizes[2], self.features_sizes[1])
        self.fc2_dec_bn = nn.BatchNorm1d(self.features_sizes[1])
        self.fc3_dec = nn.Linear(self.features_sizes[1], self.features_sizes[0])
        self.fc3_dec_bn = nn.BatchNorm1d(self.features_sizes[0])
        self.fc4_dec = nn.Linear(self.features_sizes[0],self.features_sizes[3])
        self.fc4_dec_bn = nn.BatchNorm1d(self.features_sizes[3])
        self.fc5_dec = nn.Linear(self.features_sizes[0],n_classes)
        self.fc5_dec_bn = nn.BatchNorm1d(n_classes)
        self.apply(self.weight_init)

    def _get_sizes(self):
        v = torch.zeros((1, self.input_channels,
                         self.patch_size, self.patch_size))
                         
        _, c, w, h = v.size()
        size3 = c * w * h
        
        
        v = F.relu(self.conv1_bn(self.conv1(v)))
        _, c, w, h = v.size()
        size0 = c * w * h

        v = self.pool1(v)
        _, c, w, h = v.size()
        size1 = c * w * h

        v = self.conv1_bn(v)
        _, c, w, h = v.size()
        size2 = c * w * h

        return size0, size1, size2 , size3

    def forward(self,v):
       
        # forward
        #x = x.squeeze(dim=1)
        v = v.squeeze(dim=1)
        
        
        #opou x ta labeled    opou v ta unlabeled
          
       
        
        
        
        
        # Clean path gia v
        z1 = self.conv1_bn(self.conv1(v))
        
        
        z2 = self.pool1(z1)
        
        z3 = F.relu(z2).view(-1, self.features_sizes[2])
    
        y = F.relu(self.fc_enc(z3))
        
          # Dirty path gia v
        vn = v + self.s2*(self.r(*v.size())).cuda()
        z1n = self.conv1_bn(self.conv1(vn))
        z1n = z1n + self.s2*(self.r(*z1n.size())).cuda()
        z2n = self.pool1(z1n)
        z2n = z2n + self.s2*(self.r(*z2n.size())).cuda()
        z3n = F.relu(z2n).view(-1, self.features_sizes[2])
        z3n = z3n + self.s2*(self.r(*z3n.size())).cuda()
        
        q = self.fc_enc(z3n)
        
        # Decoder path     gia v  
        z3d = F.relu(self.fc1_dec_bn(self.fc1_dec(q) + z3n.view(-1, self.features_sizes[2])))
        z2d = F.relu(self.fc2_dec_bn(self.fc2_dec(z3d) + z2n.view(-1, self.features_sizes[1])))
        z1d = F.relu(self.fc3_dec_bn(self.fc3_dec(z2d) + z1n.view(-1, self.features_sizes[0])))
        nd = F.relu(self.fc4_dec_bn(self.fc4_dec(z1d) + vn.view(-1, self.features_sizes[3])))
        
        C0 = (v.view(-1, self.features_sizes[3]), nd)
        C1 = (z1.view(-1, self.features_sizes[0]), z1d)
        C2 = (z2.view(-1, self.features_sizes[1]), z2d)
        C3 = (z3.view(-1, self.features_sizes[2]), z3d)
        
      
     
        
        
        
        
      
         

         

        

        
        
        return q,y, C0, C1, C2, C3
        

class BoulchEtAl(nn.Module):
    """
    Autoencodeurs pour la visualisation d'images hyperspectrales
    A.Boulch, N. Audebert, D. Dubucq
    GRETSI 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, planes=16):
        super(BoulchEtAl, self).__init__()
        self.input_channels = input_channels
        self.aux_loss_weight = 0.1
        
        encoder_modules = []
        n = input_channels
        with torch.no_grad():
            x = torch.zeros((10, 1, self.input_channels))
            print(x.size())
            while(n > 1):
                print("---------- {} ---------".format(n))
                if n == input_channels:
                    p1, p2 = 1, 2 * planes
                elif n == input_channels // 2:
                    p1, p2 = 2 * planes, planes
                else:
                    p1, p2 = planes, planes
                encoder_modules.append(nn.Conv1d(p1, p2, 3, padding=1))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.MaxPool1d(2))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.ReLU(inplace=True))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.BatchNorm1d(p2))
                x = encoder_modules[-1](x)
                print(x.size())
                n = n // 2

            encoder_modules.append(nn.Conv1d(planes, 3, 3, padding=1))
        encoder_modules.append(nn.Tanh())
        self.encoder = nn.Sequential(*encoder_modules)
        self.features_sizes = self._get_sizes()

        self.classifier = nn.Linear(self.features_sizes, n_classes)
        self.regressor = nn.Linear(self.features_sizes, input_channels)
        self.apply(self.weight_init)

    def _get_sizes(self):
        with torch.no_grad():
            x = torch.zeros((10, 1, self.input_channels))
            x = self.encoder(x)
            _, c, w = x.size()
        return c*w

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.view(-1, self.features_sizes)
        x_classif = self.classifier(x)
        x = self.regressor(x)
        return x_classif, x


class MouEtAl(nn.Module):
    """
    Deep recurrent neural networks for hyperspectral image classification
    Lichao Mou, Pedram Ghamisi, Xiao Xang Zhu
    https://ieeexplore.ieee.org/document/7914752/
    """
    @staticmethod
    def weight_init(m):
 # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU)):
            init.uniform_(m.weight.data, -0.1, 0.1)
            init.uniform_(m.bias.data, -0.1, 0.1)

    def __init__(self, input_channels, n_classes):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(MouEtAl, self).__init__()
        self.input_channels = input_channels
        self.gru = nn.GRU(1, 64, 1, bidirectional=False) # TODO: try to change this ?
        self.gru_bn = nn.BatchNorm1d(64*input_channels)
        #self.tanh = nn.Tanh()
        self.tanh = PRTanh()
        self.fc = nn.Linear(64*input_channels, n_classes)
        self.prelu = nn.PReLU()

    def forward(self, x):
        print (x.shape)
        x = x.squeeze()
        x = x.unsqueeze(0)
        # x is in 1, N, C but we expect C, N, 1 for GRU layer
        x = x.permute(2, 1,0)
        x = self.gru(x)[0]
        # x is in C, N, 64, we permute back
        x = x.permute(1,2,0).contiguous()
        x = x.view(x.size(0), -1)
        #print (x)
        x = self.gru_bn(x)
        #print (x)
        x = self.tanh(x)
        x = self.fc(x)
        return x


def train_liu(net, optimizer, criterion, data_loader, epoch, val_loader,klepsia,strain_loader,
          display_iter=100, device=torch.device('cpu'), display=None, scheduler=None,
           supervision='full'):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi'
    """


    


    supervision='full'

   

    net.to(device)

    save_epoch = 1 if epoch > 20 else 1


    
    
    loss_win, acc_win, = None, None
    
    trl = []
    tra = []
    vll = []
    vla =[]
    timer=[]
    epoxh=[]
    
    ignored_labels = data_loader.dataset.ignored_labels
    accuracy, total = 0., 0.
    epoxh_metr=1
    

    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        # Set the network to training mode
        net.train()
        accuracy=0.
        total=0.
        train_loss = 0.
        tic = time.clock()
        tim=0.

        dataloader_iterator = iter(strain_loader)        
        # Run the training loop for one epoch
        for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            #print (target.shape,target)
            
            try:
                (data1, target1) = next(dataloader_iterator)
                
            except StopIteration:
                dataloader_iterator = iter(strain_loader)
                (data1, target1) = next(dataloader_iterator)
        
        
        
        
            
            net.train()
            
            data, target = data.to(device), target.to(device)
            data1, target1 = data1.to(device), target1.to(device)
            #print (data.shape,data1.shape,target.shape,target1.shape)
            #torch.autograd.set_detect_anomaly(True)
            # CUDA_LAUNCH_BLOCKING=1
            optimizer.zero_grad()           
          
            
            outs = net(data)
            outs1 = net(data1)
                
            doutput,coutput ,c0,c1,c2,c3 = outs
            doutput1,coutput1 ,c01,c11,c21,c31 = outs1
            
            #tsiro
           # loss = criterion[0](coutput, target) + criterion[0](doutput, target)  + 10*criterion[1](c0[0],c0[1])  + criterion[1](c1[0],c1[1])  + 0.1*criterion[1](c2[0],c2[1])  + 0.1*criterion[1](c3[0],c3[1]) + 10*criterion[1](c01[0],c01[1])  + criterion[1](c11[0],c11[1])  + 0.1*criterion[1](c21[0],c21[1])  + 0.1*criterion[1](c31[0],c31[1])
          
            
            # loss_test =  criterion[1](c0[0],c0[1])  + criterion[1](c1[0],c1[1])  + criterion[1](c2[0],c2[1])  + criterion[1](c3[0],c3[1])
            
            
            
            loss_sup = criterion[0](doutput, target) + criterion[0](coutput,target)
            loss_unsup =criterion[1](c01[0],c01[1])  + criterion[1](c11[0],c11[1])  + criterion[1](c21[0],c21[1])  + criterion[1](c31[0],c31[1])+ criterion[1](c0[0],c0[1])  + criterion[1](c1[0],c1[1])  + criterion[1](c2[0],c2[1])  + criterion[1](c3[0],c3[1])
            
            loss=loss_sup  + loss_unsup
            
            loss.backward()
            optimizer.step()

            lossa = criterion[0](coutput, target)




            train_loss += lossa.item()
            toc = time.clock()
            net.eval()
            _, coutput = torch.max(coutput, dim=1)           
            for out, pred in zip(coutput.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
                    
            
            tim+=toc-tic



















        
        del(doutput,coutput ,c0,c1,c2,c3,doutput1,coutput1 ,c01,c11,c21,c31,loss)
        
       
       
    
    
    # Update the scheduler
        train_loss /= len(data_loader)
        toc = time.clock()
        tim+=toc-tic
        
        
        if total!=0:
            train_acc=accuracy / total
        else:
            train_acc=0.
        val_acc,val_loss=val(net,val_loader,device,supervision,criterion)
      
        
        trl.append(train_loss)
        tra.append(train_acc)
        vll.append(val_loss)
        vla.append(val_acc)
        timer.append(tim)
        epoxh.append(epoxh_metr)
        loss_win = display.line(Y=np.column_stack(((np.array(trl)),np.array(vll))),
                                      X=np.column_stack((np.arange(len(trl)),np.arange(len(vll)))),
                                      win=loss_win,
                                      opts={'title': "loss",
                                            'xlabel': "Epochs",
                                            'ylabel': "Loss"
                                            })
    
        
        
        acc_win = display.line(Y=np.column_stack(((np.array(tra)),np.array(vla))),
                                      X=np.column_stack((np.arange(len(tra)),np.arange(len(vla)))),
                                      win=acc_win,
                                      opts={'title': "Acc",
                                            'xlabel': "Epochs",
                                            'ylabel': "Acc"
                                            })
        
        
        
        
        
        
        
        
        
        if val_loader is not None:
            
            metric = -val_acc
        else:
            metric = train_loss


        excel=np.column_stack((trl,tra,vll,vla,timer,epoxh))
        excel_dir=((net.__class__.__name__)+(str(klepsia))+'.csv')


        np.savetxt((excel_dir),excel,delimiter=',',fmt='%1.3f')
        
        
        epoxh_metr=epoxh_metr+1
    
            
       

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()

        # Save the weights
        if e % save_epoch == 0:
            save_model(klepsia,net, camel_to_snake(str(net.__class__.__name__)), data_loader.dataset.name, epoch=e, metric=abs(metric))








def save_model(klepsia,model, model_name, dataset_name, **kwargs):
    if klepsia==True:
         
        model_dir = './checkpoints/' + model_name + "/" + dataset_name + "/" + str(klepsia)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        if isinstance(model, torch.nn.Module):
            filename = str(datetime.datetime.now()) + "_epoch{epoch}_{metric:.2f}".format(**kwargs)
            tqdm.write("Saving neural network weights in {}".format(filename))
            torch.save(model.state_dict(), model_dir + filename + '.pth')
        else:
            filename = str(datetime.datetime.now())
            tqdm.write("Saving model params in {}".format(filename))
            joblib.dump(model, model_dir + filename + '.pkl')


def test(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams['patch_size']
    center_pixel = hyperparams['center_pixel']
    batch_size, device = 2056, hyperparams['device']
    n_classes = hyperparams['n_classes']

    kwargs = {'step': hyperparams['test_stride'], 'window_size': (patch_size, patch_size)}
    probs = np.zeros(img.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(grouper(batch_size, sliding_window(img, **kwargs)),
                      total=(iterations),
                      desc="Inference on the image"
                      ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            output = net(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu')

            if patch_size == 1 or center_pixel:
                output = output.numpy()
                
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x:x + w, y:y + h] += out
    return probs

def val(net, data_loader, device, supervision,criterion):
# TODO : fix me using metrics()
    accuracy, total = 0., 0.
    avg_loss=0.
    ignored_labels = data_loader.dataset.ignored_labels
    net.eval()
    for batch_idx, (data, target) in enumerate(data_loader):
        
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            # print (data.shape,target.shape)
            
            if supervision == 'full':
                outs = net(data)
                doutput,output,c0,c1,c2,c3 = outs 
               
                
                loss = criterion[0](output, target) 
            
            

            avg_loss += loss.item()
            _, output = torch.max(output, dim=1)
           
            for out, pred in zip(output.view(-1), target.view(-1)):
                
                if out.item() in ignored_labels:
                    
                    continue
                else:
                    
                    accuracy += out.item() == pred.item()
                    total += 1
                    
    avg_loss /= len(data_loader)
    if total!=0:
        return accuracy / total,avg_loss
    else:
        return 0.,avg_loss
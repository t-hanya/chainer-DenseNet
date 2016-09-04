# -*- coding: utf-8 -*-

import math

import chainer 
import chainer.functions as F
import chainer.links as L


class DenseBlock(chainer.Chain):
    """DenseBlock Layer"""
    
    def __init__(self, in_channels, n_layers, growth_rate, 
                 dropout_ratio=None):

        super(DenseBlock, self).__init__()

        self._layers = []
        sum_channels = in_channels
        for l in range(n_layers):
            conv = L.Convolution2D(sum_channels, growth_rate, 3, pad=1, 
                                   wscale=math.sqrt(2))
            norm = L.BatchNormalization(sum_channels)
            self.add_link('conv{}'.format(l + 1), conv)
            self.add_link('norm{}'.format(l + 1), norm)
            self._layers.append((conv, norm))
            sum_channels += growth_rate

        self.add_persistent('dropout_ratio', dropout_ratio)
    
    def __call__(self, x, test=True):
        h_all = x
        for conv, norm in self._layers:
            h = conv(F.relu(norm(h_all, test=test)))
            if self.dropout_ratio:
                h = F.dropout(h, ratio=self.dropout_ratio, train=not test)
            h_all = F.concat((h_all, h))

        return h_all


class TransitionLayer(chainer.Chain):
    """Transition Layer"""
    
    def __init__(self, in_channels, out_channels, dropout_ratio=None):
        super(TransitionLayer, self).__init__(
            norm=L.BatchNormalization(in_channels),
            conv=L.Convolution2D(in_channels, out_channels, 3, pad=1, 
                                 wscale=math.sqrt(2)),
        )
        self.add_persistent('dropout_ratio', dropout_ratio)
        
    def __call__(self, x, test=True):
        h = self.conv(F.relu(self.norm(x, test=test)))
        if self.dropout_ratio:
            h = F.dropout(h, ratio=self.dropout_ratio, train=not test)
        h = F.average_pooling_2d(h, 2, stride=2)
        return h


class DenseNet(chainer.Chain):
    """Densely Connected Convolutional Networks
    
    see: https://arxiv.org/abs/1608.06993"""
    
    def __init__(self, depth=40, growth_rate=12, in_channels=16, 
                 dropout_ratio=0.2, n_class=10):
        
        assert (depth - 4) % 3 == 0
        n_layers = int((depth - 4) / 3)
        n_ch = [in_channels + growth_rate * n_layers * i 
                    for i in range(4)]
        dropout_ratio = dropout_ratio if dropout_ratio > 0 else None

        super(DenseNet, self).__init__(
            conv0=L.Convolution2D(3, n_ch[0], 3, pad=1),
            dense1=DenseBlock(
                n_ch[0], n_layers, growth_rate, dropout_ratio),
            trans1=TransitionLayer(n_ch[1], n_ch[1], dropout_ratio),
            dense2=DenseBlock(
                n_ch[1], n_layers, growth_rate, dropout_ratio),
            trans2=TransitionLayer(n_ch[2], n_ch[2], dropout_ratio),
            dense3=DenseBlock(
                n_ch[2], n_layers, growth_rate, dropout_ratio),
            norm4=L.BatchNormalization(n_ch[3]),
            fc4=L.Linear(n_ch[3], n_class),
        )
        self._train = True

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        self._train = value
    
    def __call__(self, x):
        test = not self.train
        h = self.conv0(x)
        h = self.dense1(h, test=test)
        h = self.trans1(h, test=test)
        h = self.dense2(h, test=test)
        h = self.trans2(h, test=test)
        h = self.dense3(h, test=test)
        h = F.relu(self.norm4(h, test=test))
        h = F.average_pooling_2d(h, 8)   # input image size: 32x32
        h = self.fc4(h)
        return h

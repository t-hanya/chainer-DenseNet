#!/usr/bin/env python
# -*- coding: utf-8 -*_
"""
Training script of DenseNet on CIFAR-10 dataset.
"""


import argparse
import math

import chainer
from chainer.datasets import cifar
import chainer.links as L
from chainer import training
from chainer.training import extensions

from extension import StepShift
from model import DenseNet


def main():
    # define options
    parser = argparse.ArgumentParser(
        description='Training script of DenseNet on CIFAR-10 dataset')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Validation minibatch size')
    parser.add_argument('--numlayers', '-L', type=int, default=40,
                        help='Number of layers')
    parser.add_argument('--growth', '-G', type=int, default=12,
                        help='Growth rate parameter')
    parser.add_argument('--dropout', '-D', type=float, default=0.2,
                        help='Dropout ratio')
    args = parser.parse_args()

    # load dataset
    train, test = cifar.get_cifar10()
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # setup model
    model = L.Classifier(DenseNet(args.numlayers, args.growth, 16,
                                  args.dropout, 10))

    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    eval_model = model.copy()
    eval_model.train = False

    # setup optimizer
    optimizer = chainer.optimizers.NesterovAG(lr=0.1, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

    # setup trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, eval_model,
                                        device=args.gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))
    trainer.extend(extensions.snapshot_object(
                   model, 'model_{.updater.epoch}.npz'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())

    # devide lr by 10 at 0.5, 0.75 fraction of total number of training epochs
    iter_per_epoch = math.ceil(len(train) / args.batchsize)
    n_iter1 = int(args.epoch * 0.5 * iter_per_epoch)
    n_iter2 = int(args.epoch * 0.75 * iter_per_epoch)
    shifts = [(n_iter1, 0.01), (n_iter2, 0.001)]
    trainer.extend(StepShift('lr', shifts, optimizer))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # start training
    trainer.run()

if __name__ == '__main__':
    main()

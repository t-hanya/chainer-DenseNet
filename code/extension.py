# -*- coding: utf-8 -*-
from chainer.training import extension


class LearningRateDrop(extension.Extension):
    """Trainer extension to drop learning rate."""

    def __init__(self, drop_ratio, attr='lr', optimizer=None):
        self._drop_ratio = drop_ratio
        self._attr = attr
        self._optimizer = optimizer

    def __call__(self, trainer):
        opt =  self._optimizer or trainer.updater.get_optimizer('main')

        lr = getattr(opt, self._attr)
        lr *= self._drop_ratio
        setattr(opt, self._attr, lr)

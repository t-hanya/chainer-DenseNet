# -*- coding: utf-8 -*-
from chainer.training import extension
from chainer.training import extensions


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


class Evaluator(extensions.Evaluator):
    """Extension of chainer.extentions.Evaluator to set train flag"""

    def evaluate(self):
        target = self._targets['main']
        target.predictor.train = False
        result = super(Evaluator, self).evaluate()
        target.predictor.train = True

        return result

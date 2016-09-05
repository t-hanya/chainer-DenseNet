# -*- coding: utf-8 -*-
from chainer.training import extension


class StepShift(extension.Extension):

    """Trainer extention to change an optimizer attribute on specified timing.

    Args:
        attr (str): Name of the attribute to change:
        shifts (list of tuple(int, float)): List of pairs of timing and target
            value. Timing is specified as count of calls.
        optimizer (~chainer.Optimizer): Target optimizer object. If it is None,
            the main optimizer of the trainer is used.
    """

    invoke_before_training = True

    def __init__(self, attr, shifts, optimizer=None):
        self._attr = attr
        self._shifts = shifts
        self._optimizer = optimizer
        self._t = 0

    def __call__(self, trainer):
        for time, value in self._shifts:
            if self._t == time:
                optimizer = self._optimizer
                if not optimizer:
                    optimizer = trainer.updater.get_optimizer('main')
                setattr(optimizer, self._attr, value)
        self._t += 1

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)

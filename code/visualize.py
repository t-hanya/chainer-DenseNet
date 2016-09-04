#!/usr/env/bin python
# -*- coding: utf-8 -*-
"""
Visualize training result.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    # define command line argument
    parser = argparse.ArgumentParser(description='Visualize training result')
    parser.add_argument('log', help='log file path to visuzlize')
    parser.add_argument('--out', '-o', default='.', help='output directory')
    args = parser.parse_args()

    # prepare output directory
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    # visualize training loss
    data = json.load(open(args.log))
    epoch = np.array([d["epoch"] for d in data])
    training_loss = np.array([d["main/loss"] for d in data])

    plt.plot(epoch, training_loss)
    plt.yscale('log')
    plt.ylabel('training loss')
    plt.xlabel('epoch')
    plt.grid(True)

    png_path = os.path.join(args.out, 'training_loss.png')
    plt.savefig(png_path)
    plt.close()

    # visualize test error
    test_acc = np.array([d["validation/main/accuracy"] for d in data])
    test_error = (1. - test_acc) * 100

    plt.plot(epoch, test_error)
    plt.yscale('linear')
    plt.ylabel('test_error [%]')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.ylim([0, 20])

    png_path = os.path.join(args.out, 'test_error.png')
    plt.savefig(png_path)
    plt.close()

if __name__ == '__main__':
    main()

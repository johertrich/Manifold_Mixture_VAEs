#!/usr/bin/env python

# This file contains general utility functions. So far only and graph utils.

import torch
import numpy as np
import copy
import sys

def get_device():
    cpu=False
    cuda=False
    for arg in sys.argv[1:]:
        if arg=='--cpu':
            cpu=True
        if arg=='--cuda':
            cuda=True
    if cpu and cuda:
        raise NameError('Use either cpu or cuda!')
    if cpu:
        return 'cpu'
    if cuda:
        return 'cuda'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

def get_base_dir():
    return '.'


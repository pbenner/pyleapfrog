## Copyright (c) 2021-2022 Philipp Benner
## 
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
## 
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
## 
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.

import numpy as np
import torch

from typing import List

## ----------------------------------------------------------------------------

from leapfrog_util import _leapfrog_regularize

## Leapfrog parameters
## ----------------------------------------------------------------------------

class Parameter(torch.nn.Parameter):
    def __new__(cls, data, q: List[int], unique=False, weight_decay=None):
        if q is not None and len(q) == 0:
            raise ValueError
        if q is None and weight_decay is None:
            raise ValueError
        if weight_decay is not None and len(weight_decay) != data.size(0):
            raise ValueError
        x          = torch.nn.Parameter.__new__(cls, data=data)
        x.q        = q
        x.data_old = torch.zeros (x.shape)
        x.nu       = np.zeros(x.shape[1], dtype=np.float32)
        x.sigma    = np.zeros(x.shape[1], dtype=np.float32)
        if weight_decay is None:
            x.weight_decay = np.zeros(data.size(0), dtype=np.float32)
        else:
            x.weight_decay = weight_decay
        if unique:
            x.exclude = np.array(x.shape[1]*[False])
        else:
            x.exclude = None
        return x

## Leapfrog linear layer
## ----------------------------------------------------------------------------

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, q: int, unique=False, weight_decay=None, bias=True):
        super().__init__()
        self.module        = torch.nn.Linear(in_features, out_features, bias=bias)
        self.module.weight = Parameter(self.module.weight, q, unique=unique, weight_decay=weight_decay)
    def forward(self, *args, **kwargs):
        # Simply forward and args and kwargs to module
        return self.module(*args, **kwargs)
    def selected_features(self):
        return list(torch.where(self.module.weight != 0.0)[1].numpy())

## Leapfrog optimizer
## ----------------------------------------------------------------------------

def one_smaller(a: torch.Tensor, max):
    r = 0.0
    for value in a:
        if value > r and value < max:
            r = value
    return value

class Optimizer():
    def __init__(self, optimizer, tolerance=1e-4):
        self.optimizer = optimizer
        self.loss      = None
        self.tolerance = tolerance

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self, *args, **kwargs):
        return self.optimizer.state_dict(*args, **kwargs)

    def step(self):
        ## copy data to data_old
        for param_group in self.optimizer.param_groups:
            for parameters in param_group['params']:
                if isinstance(parameters, Parameter):
                    parameters.data_old.copy_(parameters.data)
        ## make gradient step
        self.optimizer.step()
        ## regularize result
        for param_group in self.optimizer.param_groups:
            for parameters in param_group['params']:
                if isinstance(parameters, Parameter):
                    if parameters.exclude is not None:
                        parameters.exclude.fill(False)
                    for i in range(0, parameters.data.size(0)):
                        self._leapfrog_regularize(parameters, i)

    def _leapfrog_regularize(self, parameters, i):
        if parameters.q is not None:
            if parameters.q[0] is None:
                # do not regularize
                return
            if parameters.q[0] < 0 or parameters.q[0] >= parameters.data[i].size(0):
                # do not regularize
                return
        parameters.weight_decay[i] = _leapfrog_regularize(
            parameters.data    [i].numpy(),
            parameters.data_old[i].numpy(),
            parameters.grad    [i].numpy(),
            parameters.nu,
            parameters.sigma,
            parameters.exclude,
            parameters.q[0])

    def converged(self, loss):
        converged = False
        if self.loss is not None:
            converged = abs(loss - self.loss) < self.tolerance
        self.loss = loss
        return converged

    def get_weight_decay(self):
        weight_decay = []
        for param_group in self.optimizer.param_groups:
            for parameters in param_group['params']:
                if isinstance(parameters, Parameter):
                    weight_decay.append(parameters.weight_decay)
        return weight_decay

    def get_q(self):
        q = []
        for param_group in self.optimizer.param_groups:
            for parameters in param_group['params']:
                if isinstance(parameters, Parameter):
                    q.append(parameters.q[0])
        return q

    def next_target(self):
        finished = True
        for param_group in self.optimizer.param_groups:
            for parameters in param_group['params']:
                if isinstance(parameters, Parameter):
                    if len(parameters.q) > 1:
                        parameters.q = parameters.q[1:]
                        finished = False
        return finished

## Leapfrog penalty loss
## ----------------------------------------------------------------------------

def loss(model):
    loss = 0.0
    for parameters in model.parameters():
        if isinstance(parameters, Parameter):
            for i in range(0, parameters.data.size(0)):
                loss += (parameters.weight_decay[i]*torch.sum(torch.abs(parameters.data[i]))).item()
    return loss

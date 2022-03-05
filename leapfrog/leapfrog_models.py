## Copyright (c) 2022 Philipp Benner
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

import torch

from .leapfrog import Linear

## Leapfrog logistic regression
## ----------------------------------------------------------------------------

# Definition of a simple logistic regression model, where
# the weights are subject to leapfrog regularization
class LogisticModel(torch.nn.Module):
    def __init__(self, p, q, weight_decay=None):
        super(LogisticModel, self).__init__()
        # The Leapfrog linear layer is identical to the Torch
        # linear layer, except that the weights are subject to
        # leapfrog regularization
        self.linear = Linear(p, 1, q, weight_decay=weight_decay)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x

    def predict(self, X, device=None):
        X = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            y_hat = self(X)
        return y_hat.cpu().numpy()

## Leapfrog neural network
## ----------------------------------------------------------------------------

class LeapfrogModel(torch.nn.Module):
    def __init__(self, p, q, ks, q_steps=[], weight_decay=None, skip_connections=False, proxop=None, activation=torch.nn.LeakyReLU(), debug=0, seed=None):
        if seed is not None:
            torch     .manual_seed(seed)
            torch.cuda.manual_seed(seed)
        super(LeapfrogModel, self).__init__()
        self.linear      = []
        self.linear_in   = Linear(p, ks[0], q_steps+[q], independent=False, unique=False, weight_decay=weight_decay, proxop=proxop, debug=debug)
        self.linear_out  = torch.nn.Linear(ks[-1], 1)
        self.activation  = activation
        for i in range(len(ks)-1):
            self.linear.append(torch.nn.Linear(ks[i], ks[i+1]))
        self.skip_connections = skip_connections

    def block(self, x, i):
        y = self.linear[i](x)
        y = self.activation(y)
        if self.skip_connections and x.shape == y.shape:
            y = y + x
        return y

    def forward(self, x):
        x = self.linear_in(x)
        x = self.activation(x)
        for i in range(len(self.linear)):
            x = self.block(x, i)
        x = self.linear_out(x)
        return x

    def predict(self, X, device=None):
        X = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            y_hat = self(X)
        return y_hat.cpu().numpy()

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        # super().to() doesn't recognize lists of parameters... 
        for i, _ in enumerate(self.linear):
            self.linear[i] = self.linear[i].to(*args, **kwargs)
        return self

## ----------------------------------------------------------------------------
## ----------------------------------------------------------------------------

class LeapfrogIndependentModel(torch.nn.Module):
    def __init__(self, p, q, ks, q_steps=[], weight_decay=None, skip_connections=False, proxop=None, activation=torch.nn.LeakyReLU(), debug=0, seed=None):
        if seed is not None:
            torch     .manual_seed(seed)
            torch.cuda.manual_seed(seed)
        super(LeapfrogIndependentModel, self).__init__()
        self.linear_lf   = Linear(p, q, q_steps+[1], independent=True, unique=True, weight_decay=weight_decay, proxop=proxop, debug=debug)
        self.linear      = []
        self.linear_in   = torch.nn.Linear(q, ks[0])
        self.linear_out  = torch.nn.Linear(ks[-1], 1)
        self.activation  = activation
        for i in range(len(ks)-1):
            self.linear.append(torch.nn.Linear(ks[i], ks[i+1]))
        self.skip_connections = skip_connections

    def block(self, x, i):
        y = self.linear[i](x)
        y = self.activation(y)
        if self.skip_connections and x.shape == y.shape:
            y = y + x
        return y

    def forward(self, x):
        x = self.linear_lf(x)
        x = self.linear_in(x)
        for i in range(len(self.linear)):
            x = self.block(x, i)
        x = self.linear_out(x)
        return x

    def predict(self, X, device=None):
        X = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            y_hat = self(X)
        return y_hat.cpu().numpy()

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        # super().to() doesn't recognize lists of parameters... 
        for i, _ in enumerate(self.linear):
            self.linear[i] = self.linear[i].to(*args, **kwargs)
        return self

## ----------------------------------------------------------------------------
## ----------------------------------------------------------------------------

class LeapfrogRepeatModel(torch.nn.Module):
    def __init__(self, p, q, ks, q_steps=[], weight_decay=None, skip_connections=False, activation=torch.nn.LeakyReLU(), debug=0, seed=None):
        if seed is not None:
            torch     .manual_seed(seed)
            torch.cuda.manual_seed(seed)
        super(LeapfrogRepeatModel, self).__init__()
        self.linear_lf   = Linear(p, 1, q_steps+[q], independent=False, unique=False, weight_decay=weight_decay, debug=debug)
        self.linear_lf_k = ks[0]
        self.linear      = []
        self.linear_out  = torch.nn.Linear(ks[-1], 1)
        self.activation  = activation
        for i in range(len(ks)-1):
            self.linear.append(torch.nn.Linear(ks[i], ks[i+1]))
        self.skip_connections = skip_connections

    def block(self, x, i):
        y = self.linear[i](x)
        y = self.activation(y)
        if self.skip_connections and x.shape == y.shape:
            y = y + x
        return y

    def forward(self, x):
        x = self.linear_lf(x)
        x = x.repeat_interleave(self.linear_lf_k, dim=1)
        for i in range(len(self.linear)):
            x = self.block(x, i)
        x = self.linear_out(x)
        return x

    def predict(self, X, device=None):
        X = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            y_hat = self(X)
        return y_hat.cpu().numpy()

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        # super().to() doesn't recognize lists of parameters... 
        for i, _ in enumerate(self.linear):
            self.linear[i] = self.linear[i].to(*args, **kwargs)
        return self

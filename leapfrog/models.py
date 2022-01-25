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
        self.linear = lf.Linear(p, 1, q, weight_decay=weight_decay)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x

## Leapfrog neural network
## ----------------------------------------------------------------------------

class LeapfrogNeuralModel(torch.nn.Module):
    def __init__(self, p, q, ks, weight_decay=None, batch_normalization=False, dropout=None, skip_connections=False):
        torch     .manual_seed(1)
        torch.cuda.manual_seed(1)
        super(LeapfrogNeuralModel, self).__init__()
        self.linear_lf   = lf.Linear(p, q, [None, 1], unique=True, weight_decay=weight_decay)
        self.linear      = []
        self.batchnorm   = []
        self.linear_in   = torch.nn.Linear(q, ks[0])
        self.linear_out  = torch.nn.Linear(ks[-1], 1)
        self.activation  = torch.nn.ELU()
        if dropout is None:
            self.dropOut = None
        else:
            self.dropOut = torch.nn.Dropout(0.1)
        for i in range(len(ks)-1):
            self.linear   .append(torch.nn.Linear(ks[i], ks[i+1]))
            if batch_normalization:
                self.batchnorm.append(torch.nn.BatchNorm1d(ks[i+1]))
        self.skip_connections = skip_connections

    def forward(self, x):
        x = self.linear_lf(x)
        x = self.activation(x)
        x = self.linear_in(x)
        x = self.activation(x)
        # Apply dense layers
        for i in range(len(self.linear)):
            if self.dropOut is not None:
                x  = self.dropOut(x)
            x_ = self.linear[i](x)
            x_ = self.activation(x_)
            if self.skip_connections and (self.linear[i].in_features == self.linear[i].out_features):
                x = x_ + x
            else:
                x = x_
            if len(self.batchnorm) > 0:
                x = self.batchnorm[i](x)
        x = self.linear_out(x)
        return x

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            y_hat = torch.flatten(self(X))
        return y_hat.numpy()

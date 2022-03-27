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
    def __init__(self, p, q):
        super(LogisticModel, self).__init__()
        # The Leapfrog linear layer is identical to the Torch
        # linear layer, except that the weights are subject to
        # leapfrog regularization
        self.linear = Linear(p, 1, q)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x

    def predict(self, X, device=None):
        X = torch.tensor(X, dtype=torch.float32, device=device)
        self = self.to(X.device)
        with torch.no_grad():
            y_hat = self(X)
        return y_hat.cpu().numpy()

## Leapfrog ensemble
## ----------------------------------------------------------------------------

class LeapfrogEnsemble(torch.nn.Module):
    def __init__(self, models):
        super(LeapfrogEnsemble, self).__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        n = len(self.models)
        y = self.models[0].forward(x)/n
        for model in self.models[1:]:
            y += model.forward(x)/n
        return y

    def predict(self, *args, **kwargs):
        n = len(self.models)
        y = self.models[0].predict(*args, **kwargs)/n
        for model in self.models[1:]:
            y += model.predict(*args, **kwargs)/n
        return y

## Leapfrog neural network
## ----------------------------------------------------------------------------

class LeapfrogModel(torch.nn.Module):
    def __init__(self, q, ks, q_steps=[], skip_connections=False, proxop=None, activation=torch.nn.LeakyReLU(), activation_out=None, debug=0, seed=None):
        if len(ks) < 2:
            raise ValueError("invalid argument: ks must have at least two values for input and output") 
        if seed is not None:
            torch     .manual_seed(seed)
            torch.cuda.manual_seed(seed)
        super(LeapfrogModel, self).__init__()
        self.activation       = activation
        self.activation_out   = activation_out
        self.skip_connections = skip_connections
        self.linear           = torch.nn.ModuleList([])
        self.linear.append(Linear(ks[0], ks[1], q_steps+[q], independent=False, unique=False, proxop=proxop, debug=debug))
        for i in range(1, len(ks)-1):
            self.linear.append(torch.nn.Linear(ks[i], ks[i+1]))

    def block(self, x, i):
        y = self.linear[i](x)
        y = self.activation(y)
        if x.shape == y.shape:
            if type(self.skip_connections) == int  and i % self.skip_connections == 0:
                y = y + x
            if type(self.skip_connections) == bool and self.skip_connections:
                y = y + x
        return y

    def forward(self, x):
        # Apply leapfrog layer without activation
        x = self.linear[0](x)
        # Apply innear layers
        for i in range(1, len(self.linear)-1):
            x = self.block(x, i)
        # Apply final layer if available
        if len(self.linear) > 1:
            x = self.linear[-1](x)
        # Apply output activation if available
        if self.activation_out is not None:
            x = self.activation_out(x)
        return x

    def predict(self, X, device=None):
        X = torch.tensor(X, dtype=torch.float32, device=device)
        self = self.to(X.device)
        with torch.no_grad():
            y_hat = self(X)
        return y_hat.cpu().numpy()

## ----------------------------------------------------------------------------
## ----------------------------------------------------------------------------

class LeapfrogIndependentModel(torch.nn.Module):
    def __init__(self, q, ks, q_steps=[], skip_connections=False, proxop=None, activation=torch.nn.LeakyReLU(), activation_out=None, debug=0, seed=None):
        if len(ks) < 2:
            raise ValueError("invalid argument: ks must have at least two values for input and output") 
        if seed is not None:
            torch     .manual_seed(seed)
            torch.cuda.manual_seed(seed)
        super(LeapfrogIndependentModel, self).__init__()
        self.activation       = activation
        self.activation_out   = activation_out
        self.skip_connections = skip_connections
        self.linear           = torch.nn.ModuleList([])
        self.linear.append(Linear(ks[0], ks[1], q_steps+[q], independent=True, unique=True, proxop=proxop, debug=debug))
        for i in range(1, len(ks)-1):
            self.linear.append(torch.nn.Linear(ks[i], ks[i+1]))

    def block(self, x, i):
        y = self.linear[i](x)
        y = self.activation(y)
        if x.shape == y.shape:
            if type(self.skip_connections) == int  and i % self.skip_connections == 0:
                y = y + x
            if type(self.skip_connections) == bool and self.skip_connections:
                y = y + x
        return y

    def forward(self, x):
        # Apply leapfrog layer without activation
        x = self.linear[0](x)
        # Apply innear layers
        for i in range(1, len(self.linear)-1):
            x = self.block(x, i)
        # Apply final layer if available
        if len(self.linear) > 1:
            x = self.linear[-1](x)
        # Apply output activation if available
        if self.activation_out is not None:
            x = self.activation_out(x)
        return x

    def predict(self, X, device=None):
        X = torch.tensor(X, dtype=torch.float32, device=device)
        self = self.to(X.device)
        with torch.no_grad():
            y_hat = self(X)
        return y_hat.cpu().numpy()

## ----------------------------------------------------------------------------
## ----------------------------------------------------------------------------

class LeapfrogRepeatModel(torch.nn.Module):
    def __init__(self, q, ks, q_steps=[], skip_connections=False, proxop=None, activation=torch.nn.LeakyReLU(), activation_out=None, debug=0, seed=None):
        if len(ks) < 2:
            raise ValueError("invalid argument: ks must have at least two values for input and output") 
        if seed is not None:
            torch     .manual_seed(seed)
            torch.cuda.manual_seed(seed)
        super(LeapfrogRepeatModel, self).__init__()
        self.activation       = activation
        self.activation_out   = activation_out
        self.skip_connections = skip_connections
        self.linear           = torch.nn.ModuleList([])
        self.linear_k         = ks[1]
        self.linear.append(Linear(ks[0], 1, q_steps+[q], independent=False, unique=False, proxop=proxop, debug=debug))
        for i in range(1, len(ks)-1):
            self.linear.append(torch.nn.Linear(ks[i], ks[i+1]))

    def block(self, x, i):
        y = self.linear[i](x)
        y = self.activation(y)
        if x.shape == y.shape:
            if type(self.skip_connections) == int  and i % self.skip_connections == 0:
                y = y + x
            if type(self.skip_connections) == bool and self.skip_connections:
                y = y + x
        return y

    def forward(self, x):
        # Apply leapfrog layer without activation
        x = self.linear[0](x)
        x = x.repeat_interleave(self.linear_k, dim=1)
        # Apply innear layers
        for i in range(1, len(self.linear)-1):
            x = self.block(x, i)
        # Apply final layer if available
        if len(self.linear) > 1:
            x = self.linear[-1](x)
        # Apply output activation if available
        if self.activation_out is not None:
            x = self.activation_out(x)
        return x

    def predict(self, X, device=None):
        X = torch.tensor(X, dtype=torch.float32, device=device)
        self = self.to(X.device)
        with torch.no_grad():
            y_hat = self(X)
        return y_hat.cpu().numpy()

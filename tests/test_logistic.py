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

import leapfrog as lf
import numpy as np
import torch

from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset

## Leapfrog logistic model
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

## Train logistic model
## ----------------------------------------------------------------------------

def train_logistic_model():
    # Number of samples
    n = 1000
    # Number of features
    p = 100
    # Generate features X and labels y
    X, y = make_classification(n_samples=n, n_features=p, random_state=np.random.RandomState(0))

    # Specify a list of the number of features we want to estimate
    q = [0,1,2,60]

    # Define a simple logistic regression model
    model = LogisticModel(X.shape[1], q)

    # Logistic regression models are trained using the binary cross-entropy
    loss_function = torch.nn.BCELoss()
    # We use a standard optimizer such as Adam...
    optimizer     = torch.optim.Adam(model.parameters(), lr=0.1)
    # and augment it with a Leapfrog optimizer that performs the
    # regularization steps
    optimizer     = lf.Optimizer(optimizer, tolerance=1e-5)
    # A maximum of max_epochs is used for training
    max_epochs    = 500
    # Define the training set and data loader for training, with
    # a batch size equal to the full training set (i.e. do not use
    # stochastic gradient descent, because the data set and model
    # are fairly small)
    trainloader   = DataLoader(
        TensorDataset(torch.Tensor(X), torch.Tensor(y)),
        batch_size=int(X.shape[0]),
        shuffle=False,
        num_workers=1)

    # Record the regularization strength lambda
    l_     = []
    # Record the loss
    loss_  = []
    # Record the coefficients of the logistic regression
    coefs_ = []
    while True:
        # Do a maximum of max_epochs iterations over the training set
        for _epoch in range(0, max_epochs):
            # Loop over the training set
            for i, data in enumerate(trainloader, 0):
                # Get X (inputs) and y (targets)
                inputs, targets = data
                # Reset gradient
                optimizer.zero_grad()
                # Evaluate model
                outputs = torch.flatten(model(inputs))
                # Compute loss
                loss = loss_function(outputs, targets)
                # Backpropagate gradient
                loss.backward()
                # Perform one gradient descent step
                optimizer.step()
                # Record loss
                loss_.append(loss.item())

                #print(f'Loss: {loss.item()}')

            # Check if optimizer converged (only possible of no
            # stochastic gradient descent is used), or if the
            # maximum number of epochs is reached
            if optimizer.converged(loss.item()) or _epoch == max_epochs-1:
                # Record lambda (weight decay)
                weight_decay = optimizer.get_weight_decay()
                # Each layer and output node has its own weight decay parameter. We
                # have only one layer and output node.
                l_.append(weight_decay[0][0].item())
                # Record coefficients (weights) from the linear layer
                coefs_.append(list(model.parameters())[0][0].detach().numpy().copy())
                break

        print(f'Training process has finished for target q={optimizer.get_q()[0]}.')
        # Select the next q (number of features) for optimization
        if optimizer.next_target():
            # There are no more targets, exit loop
            break

    return np.array(l_), np.array(loss_), coefs_

## Tests
## ----------------------------------------------------------------------------

def test_logistic():
    l, loss, coefs = train_logistic_model()

    # Test regularization strengths
    assert np.sum(np.abs(l - [0.4989, 0.04353, 0.03012, 0.0029])) < 1e-2, f'Invalid regularization strengths: {l}'

    # Test final loss
    assert np.abs(loss[-1] - 0.1283) < 1e-2, "Invalid final loss"

    # Test number of parameters
    assert np.sum(coefs[0] != 0.0) ==  0, "Invalid number of parameters"
    assert np.sum(coefs[1] != 0.0) ==  1, "Invalid number of parameters"
    assert np.sum(coefs[2] != 0.0) ==  2, "Invalid number of parameters"
    assert np.sum(coefs[3] != 0.0) == 60, "Invalid number of parameters"

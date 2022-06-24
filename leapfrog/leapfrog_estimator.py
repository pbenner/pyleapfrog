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

import sys
import torch

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from .leapfrog_stopper import LeapfrogStopper
from .leapfrog import Optimizer

## Leapfrog model Estimator
## ----------------------------------------------------------------------------

class LeapfrogEstimator:
    def __init__(self, model, epochs=10000, tolerance=1e-4, patience=7, warm_up_steps=100, val_size=0.0, optimizer=torch.optim.Adam, loss_function=torch.nn.L1Loss(), random_state=42, shuffle=True, batch_size=None, device=None, optimargs={}, verbose=False, verbose_grad=False):

        self.model         = model
        self.epochs        = epochs
        self.warm_up_steps = warm_up_steps
        self.patience      = patience
        self.tolerance     = tolerance
        self.val_size      = val_size
        self.shuffle       = shuffle
        self.batch_size    = batch_size
        self.optimizer     = optimizer
        self.loss_function = loss_function
        self.device        = device
        self.random_state  = random_state
        self.verbose       = verbose
        self.verbose_grad  = verbose_grad
        self.optimargs     = optimargs

    def fit(self, X, y, **kwargs):
        return self(X, y, **kwargs)

    def __call__(self, X, y, X_val=None, y_val=None, loss_function=None, device=None):

        if self.val_size > 0.0:
            assert X_val is None and y_val is None, f'val_size is non-zero and X_val / y_val are given'
        if X_val is None:
            assert y_val is None, f'X_val is None but y_val is not'
        if X_val is not None:
            assert y_val is not None, f'X_val filled with plenty of beautiful data, but y_val is None'
        if loss_function is None:
            loss_function = self.loss_function
        if device is None:
            device = self.device
        # Bugfix when `TransformedTargetRegressor` is used, which
        # flattens the target array
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if y_val is not None:
            if len(y_val.shape) == 1:
                y_val = y_val.reshape(-1, 1)

        optimizer = self._get_optimizer()

        hist_train = []
        hist_val   = []

        if self.val_size > 0.0:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size, random_state=self.random_state, shuffle=True)

            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_val   = torch.tensor(X_val  , dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            y_val   = torch.tensor(y_val  , dtype=torch.float32)

            if device is not None:
                X_train, X_val = X_train.to(device), X_val.to(device)
                y_train, y_val = y_train.to(device), y_val.to(device)

        else:
            X_train = torch.tensor(X, dtype=torch.float32)
            y_train = torch.tensor(y, dtype=torch.float32)
            if device is not None:
                X_train = X_train.to(device)
                y_train = y_train.to(device)
            # If no validation data is given, use training data for validation
            if X_val is None:
                X_val = X_train
                y_val = y_train
            else:
                X_val = torch.tensor(X_val, dtype=torch.float32)
                y_val = torch.tensor(y_val, dtype=torch.float32)
                if device is not None:
                    X_val = X_val.to(device)
                    y_val = y_val.to(device)

        if device is not None:
            self.model = self.model.to(device)

        if self.batch_size is None:
            self.batch_size = int(X_train.shape[0])
            self.shuffle    = False

        trainloader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=self.shuffle, num_workers=0)

        es = LeapfrogStopper(patience=self.patience, verbose=self.verbose)

        final_target = optimizer.is_final()

        if self.epochs is None:
            self.epochs = sys.maxsize

        for _epoch in range(0, self.epochs):
            if self.verbose:
                print(f'\nTraining epoch: {_epoch}')
                print(f'Final target: {final_target}')
                sys.stdout.flush()
            self.model.train()
            loss_sum = 0.0
            loss_n   = 0.0
            for _, data in enumerate(trainloader, 0):
                # Get X (inputs) and y (targets)
                X_batch, y_batch = data
                # Reset gradient
                optimizer.zero_grad()
                # Evaluate model
                y_hat = self.model(X_batch)
                # Compute loss
                assert y_hat.shape == y_batch.shape, 'Internal Error'
                loss = loss_function(y_hat, y_batch)
                # Backpropagate gradient
                loss.backward()
                # Perform one gradient descent step
                optimizer.step()
                # Record loss
                loss_sum += loss.item()
                loss_n   += 1.0
            
            # Apply leapfrog regularization
            optimizer.regularize()

            loss_train = loss_sum / loss_n
            if self.verbose:
                print(f'Loss train: {loss_train}')
                sys.stdout.flush()

            # Record train loss
            hist_train.append(loss_train)

            # Get next target
            if self.warm_up_steps is not None:
                if not final_target and _epoch % self.warm_up_steps == 0:
                    es.reset_full()
                    if optimizer.next_target():
                        final_target = True
                        if self.verbose:
                            print(f'Fitting final target')
                            sys.stdout.flush()
                    else:
                        if self.verbose:
                            print(f'Fitting next target')
                            sys.stdout.flush()

            # Get validation loss
            self.model.eval()
            if X_train is X_val:
                loss_val = loss_train
            else:
                with torch.no_grad():
                    y_hat = self.model(X_val)
                    assert y_hat.shape == y_val.shape, 'Internal Error'
                    loss_val = loss_function(y_hat, y_val).item()
                # Record validation loss
                hist_val.append(loss_val)

            # If verbose print validation loss
            if self.verbose:
                print(f'Loss val  : {loss_val}')
                sys.stdout.flush()
            # Check EarlyStopping
            if es(loss_val, self.model):
                if final_target:
                    break
                else:
                    if self.warm_up_steps is None:
                        es.reset_full()
                        if optimizer.next_target():
                            final_target = True
                            if self.verbose:
                                print(f'Fitting final target')
                                sys.stdout.flush()
                        else:
                            if self.verbose:
                                print(f'Fitting next target')
                                sys.stdout.flush()

        if self.verbose and _epoch == self.epochs-1:
            print(f'Maximum number of epochs reached')
            sys.stdout.flush()

        self.model.load_state_dict(es.model)
        self.model.eval()

        if X_train is X_val:
            return {'train_loss': hist_train}
        else:
            return {'train_loss': hist_train, 'val_loss': hist_val}

    def _get_optimizer(self):
        # Get optimizer specified by the user
        optimizer = self.optimizer(self.model.parameters(), **self.optimargs)
        # and augment it with a Leapfrog optimizer that performs the
        # regularization steps
        optimizer = Optimizer(optimizer, tolerance=self.tolerance, verbose_grad=self.verbose_grad)
        return optimizer

    def get_model(self):
        return self.model

    def predict(self, *args, device=None, **kwargs):
        if device is None:
            device = self.device
        return self.model.predict(*args, device=device, **kwargs)

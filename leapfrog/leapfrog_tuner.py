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

import numpy as np
import sys
import torch

from multiprocessing.pool import ThreadPool as Pool
from sklearn.model_selection import KFold

## Leapfrog hyperparameter tuner
## ----------------------------------------------------------------------------

class LeapfrogTuner:
    def __init__(self, get_model, parameters, n_splits=10, n_jobs=10, refit=False, use_test_as_val=False, random_state=None, device=None, verbose=False):
        self.get_model       = get_model
        self.parameters      = parameters
        self.n_splits        = n_splits
        self.model           = None
        self.n_jobs          = n_jobs
        self.refit           = refit
        self.random_state    = random_state
        self.device          = device
        self.verbose         = verbose
        self.use_test_as_val = use_test_as_val

    def fit(self, X, y, loss_function=torch.nn.L1Loss(), **kwargs):
        if self.verbose:
            print(f'Testing >> {len(self.parameters)} << configurations in >> {self.n_splits} <<-fold CV:')
            sys.stdout.flush()
        # Bugfix when `TransformedTargetRegressor` is used, which
        # flattens the target array
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        with Pool(self.n_jobs) as pool:
            r = pool.map(lambda i: self._fit_cv(X, y, i, loss_function, **kwargs), range(len(self.parameters)))
        # extract performances
        error = [ d['error'] for d in r]
        # Get best parameters
        k = np.argmin(error)
        if self.verbose:
            print(f'=> Errors >> {error} << => Selecting configuration {k+1}')
            sys.stdout.flush()
        if self.refit:
            # Fit model on full data
            self.model = self.get_model(X.shape[1], self.parameters[k])
            self.model.fit(X, y, **kwargs)
        else:
            self.model = r[k]['models']

    def _fit_cv(self, X, y, k, loss_function, **kwargs):
        # This function processes one CV-fold
        def process_fold(x):
            # Unravel parameters
            (i, (i_train, i_test)) = x

            if self.verbose:
                print(f'=> Testing configuration >> {k+1} / {len(self.parameters)} << in CV step >> {i+1} / {self.n_splits} <<')
                sys.stdout.flush()

            # Set device for training. If self.device is a list, select
            # the k-th device for training
            if type(self.device) == list:
                kwargs['device'] = self.device[k % len(self.device)]
            else:
                if self.device is not None:
                    kwargs['device'] = self.device

            X_train = X[i_train]
            y_train = y[i_train]
            X_test  = X[i_test]
            y_test  = y[i_test]

            model = self.get_model(self.parameters[k])
            if self.use_test_as_val:
                model.fit(X_train, y_train, X_val=X_test, y_val=y_test, **kwargs)
            else:
                model.fit(X_train, y_train, **kwargs)

            # Evaluate model
            with torch.no_grad():
                y_hat  = model.predict(X_test)
                y_hat  = torch.tensor(y_hat , dtype=torch.float32)
                y_test = torch.tensor(y_test, dtype=torch.float32)
                assert y_hat.shape == y_test.shape, 'Internal Error'
                test_loss = loss_function(y_test, y_hat).item()

            return model, test_loss

        # Process all CV-folds
        result = map(process_fold, enumerate(KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state).split(X, y=y)))
        result = list(result)

        # Split result
        models = [ x[0] for x in result ]
        errors = [ x[1] for x in result ]

        return {'error': np.mean(errors), 'models': models}

    def predict(self, *args, **kwargs):
        if self.refit:
            return self.model.predict(*args, **kwargs)
        else:
            n = len(self.model)
            y = self.model[0].predict(*args, **kwargs)/n
            for model in self.model[1:]:
                y += model.predict(*args, **kwargs)/n
            return y

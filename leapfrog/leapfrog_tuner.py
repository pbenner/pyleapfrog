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

from random import shuffle
import numpy as np
import sys
import torch

from multiprocessing.pool import ThreadPool as Pool
from sklearn.model_selection import KFold

from .leapfrog_models import LeapfrogEnsemble

## Leapfrog hyperparameter tuner
## ----------------------------------------------------------------------------

class LeapfrogTuner:
    def __init__(self, get_estimator, parameters, n_splits=10, n_jobs=10, loss_function=torch.nn.L1Loss(), summarizer=np.mean, shuffle=True, sort_target=None, refit=False, use_test_as_val=False, random_state=None, device=None, verbose=False):
        self.get_estimator   = get_estimator
        self.parameters      = parameters
        self.n_splits        = n_splits
        self.model           = None
        self.n_jobs          = n_jobs
        self.refit           = refit
        self.random_state    = random_state
        self.device          = device
        self.verbose         = verbose
        self.use_test_as_val = use_test_as_val
        self.loss_function   = loss_function
        self.shuffle         = shuffle
        self.sort_target     = sort_target
        self.summarizer      = summarizer

    def fit(self, X, y, **kwargs):
        if self.verbose:
            print(f'Testing >> {len(self.parameters)} << configurations in >> {self.n_splits} <<-fold CV:')
            sys.stdout.flush()
        # Bugfix when `TransformedTargetRegressor` is used, which
        # flattens the target array
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        with Pool(self.n_jobs) as pool:
            r = pool.map(lambda i: self._fit_cv(X, y, i, **kwargs), range(len(self.parameters)))
        # extract performances
        error = [ d['error'] for d in r]
        # Get best parameters
        k = np.argmin(error)
        if self.verbose:
            print(f'=> Errors >> {error} << => Selecting configuration {k+1}')
            sys.stdout.flush()
        if self.refit:
            # Fit model on full data
            estimator = self.get_estimator(X.shape[1], self.parameters[k])
            estimator.fit(X, y, **kwargs)
            self.model = estimator.get_model()
        else:
            self.model = LeapfrogEnsemble(r[k]['models'])

    def _fit_cv(self, X, y, k, loss_function=None, device=None, **kwargs):
        # Get default values from constructor
        if loss_function is None:
            loss_function = self.loss_function
        if device is None:
            device = self.device
        # Set device for training. If device is a list, select
        # the k-th device for training
        if type(device) == list:
            device = device[k % len(device)]
        # Sort target values
        if self.sort_target is not None:
            i_sorted = np.argsort(y[:,self.sort_target])
            X = X[i_sorted]
            y = y[i_sorted]
        # This function processes one CV-fold
        def process_fold(x):
            # Unravel parameters
            (i, (i_train, i_test)) = x

            if self.verbose:
                print(f'=> Testing configuration >> {k+1} / {len(self.parameters)} << in CV step >> {i+1} / {self.n_splits} <<')
                sys.stdout.flush()

            X_train = X[i_train]
            y_train = y[i_train]
            X_test  = X[i_test]
            y_test  = y[i_test]

            estimator = self.get_estimator(self.parameters[k])
            if self.use_test_as_val:
                estimator.fit(X_train, y_train, X_val=X_test, y_val=y_test, **kwargs)
            else:
                estimator.fit(X_train, y_train, **kwargs)

            model = estimator.get_model()
            # Evaluate model
            with torch.no_grad():
                y_hat  = model.predict(X_test)
                y_hat  = torch.tensor(y_hat , dtype=torch.float32)
                y_test = torch.tensor(y_test, dtype=torch.float32)
                assert y_hat.shape == y_test.shape, 'Internal Error'
                test_loss = loss_function(y_test, y_hat).item()

            return model, test_loss

        # Process all CV-folds
        result = map(process_fold, enumerate(KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state).split(X, y=y)))
        result = list(result)

        # Split result
        models = [ x[0] for x in result ]
        errors = [ x[1] for x in result ]

        if self.verbose:
            print(f'=> Configuration {k+1} finished with errors: {self.summarizer(errors)}=summary({errors})')

        return {'error': self.summarizer(errors), 'models': models}

    def predict(self, *args, device=None, **kwargs):
        if device is None:
            device = self.device
        return self.model.predict(*args, device=device, **kwargs)

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

from multiprocessing.pool import ThreadPool as Pool
from sklearn.model_selection import KFold

## Leapfrog hyperparameter tuner
## ----------------------------------------------------------------------------

class LeapfrogTuner:
    def __init__(self, get_model, parameters, n_splits=10, n_jobs=10, refit=False, use_test_as_val=False, random_state=None, verbose=False):
        self.get_model       = get_model
        self.parameters      = parameters
        self.n_splits        = n_splits
        self.model           = None
        self.n_jobs          = n_jobs
        self.refit           = refit
        self.random_state    = random_state
        self.verbose         = verbose
        self.use_test_as_val = use_test_as_val

    def fit(self, X, y, **kwargs):
        if self.verbose:
            print(f'Testing >> {len(self.parameters)} << configurations in >> {self.n_splits} <<-fold CV:')
            sys.stdout.flush()
        with Pool(self.n_jobs) as p:
            r = p.map(lambda i: self._fit_cv(X, y, i, **kwargs), range(len(self.parameters)))
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

    def _fit_cv(self, X, y, i, **kwargs):
        error_fold = []
        models     = []

        # Test parameters with k-fold cross-validation
        for i, (i_train, i_test) in enumerate(
            KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state).split(X, y=y)
        ):
            if self.verbose:
                print(f'=> Testing configuration >> {i+1} / {len(self.parameters)} << in CV step >> {i+1} / {self.n_splits} <<')
                sys.stdout.flush()

            X_train = X[i_train,:]
            y_train = y[i_train]
            X_test  = X[i_test,:]
            y_test  = y[i_test]

            model = self.get_model(self.parameters[i])
            if self.use_test_as_val:
                model.fit(X_train, y_train, X_val=X_test, y_val=y_test, **kwargs)
            else:
                model.fit(X_train, y_train, **kwargs)

            # Save model error
            error_fold.append(model.evaluate(X_test, y_test))
            # Save model
            models.append(model)

        return {'error': np.mean(error_fold), 'models': models}

    def predict(self, *args, **kwargs):
        if self.refit:
            return self.model.predict(*args, **kwargs)
        else:
            n = len(self.model)
            y = self.model[0].predict(*args, **kwargs)/n
            for model in self.model[1:]:
                y += model.predict(*args, **kwargs)/n
            return y

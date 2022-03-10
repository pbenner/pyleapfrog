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

## ----------------------------------------------------------------------------

class LeapfrogStopper:
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        self.patience       = patience
        self.verbose        = verbose
        self.counter        = 0
        self.early_stop     = False
        self.val_loss_min   = None
        self.val_loss_round = None
        self.delta          = delta
        self.trace_func     = trace_func
        self.model          = None

    def __call__(self, val_loss, model):

        if np.isnan(val_loss):
            self.early_stop = True
            return self.early_stop

        if self.val_loss_min is None:
            self.val_loss_min = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.val_loss_min - self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(val_loss, model)
        
        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'EarlyStopping: Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.model          = model.state_dict()
        self.val_loss_min   = val_loss
        self.val_loss_round = val_loss
        self.counter        = 0

    def reset(self):
        self.counter        = 0
        self.early_stop     = False
        self.val_loss_round = None

    def reset_full(self):
        self.reset()
        self.val_loss_min   = None
        self.val_loss_round = None
        self.model          = None

# cython: infer_types=True
##
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

cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport abs, isnan, isinf

## ----------------------------------------------------------------------------

np.import_array()

## ----------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef np.float32_t sign(np.float32_t x):
    if x < 0.0:
        return -1.0
    if x > 0.0:
        return  1.0
    return 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef np.float32_t __compute_lambda(np.ndarray[np.float32_t, ndim=1] _sigma, Py_ssize_t n, np.float32_t l):
    cdef np.float32_t[::1] sigma = _sigma
    cdef np.float32_t r = 0.0

    for i in range(n):
        if sigma[i] > r and sigma[i] < l:
            r = sigma[i]

    return (r+l)/(<np.float32_t>2.0)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef np.float32_t __leapfrog_regularize(np.ndarray[np.float32_t, ndim=1] _data, _data_old, _grad, _nu, _sigma, np.ndarray[np.uint8_t, ndim=1] _exclude, Py_ssize_t n, q):
    cdef np.float32_t[::1] data     = _data
    cdef np.float32_t[::1] data_old = _data_old
    cdef np.float32_t[::1] grad     = _grad
    cdef np.float32_t[::1] nu       = _nu
    cdef np.float32_t[::1] sigma    = _sigma
    cdef np.uint8_t  [::1] exclude  = _exclude

    cdef np.float32_t l, v
    cdef Py_ssize_t k = 0
    cdef Py_ssize_t i

    # Compute nu and sigma
    for i in range(n):
        if isnan(data[i]):
            data    [i] = 0.0
        if isnan(data_old[i]):
            data_old[i] = 0.0

        nu   [i] = abs((data[i] - data_old[i])/grad[i])
        sigma[i] = abs(data[i])/nu[i]

        if isnan(sigma[i]) or isinf(sigma[i]):
            nu   [i] = 0.0
            sigma[i] = 0.0
        
        if exclude is not None:
            if exclude[i]:
                nu   [i] = 0.0
                sigma[i] = 0.0

    # Partially sort sigma to find the q-th largest value
    _sigma.partition(-q)
    # Get q-th largest value
    l = sigma[n-q]
    # Get next smaller element
    l = __compute_lambda(_sigma, n, l)

    # Smallest value
    v = -1.0
    # Update weights
    for i in range(n):
        if nu[i] == 0.0:
            data[i] = 0.0
            continue
        if exclude is not None:
            if exclude[i]:
                data[i] = 0.0
                continue
        if abs(data[i]) <= l*nu[i]:
            data[i] = 0.0
            continue
        # Apply proximal operator
        data[i] = sign(data[i])*(abs(data[i]) - l*nu[i])
        # Count number of non-zero parameters
        k += 1
        # Record smallest absolute value
        if v == -1.0 or abs(data[i]) < v:
            v = abs(data[i])
        # Exclude this in future steps
        if exclude is not None:
            exclude[i] = True

    # Fix issue when data contains multiple identical elements
    if k > q:
        for i in range(n):
            # Identify smallest elements in data
            if abs(data[i]) == v:
                # Set to zero
                data[i] = 0.0
                # Reduce count of non-zero elements
                k -= 1
                # Until q elements are non-zero
                if k == q:
                    break

    return l

## Wrapper
## ----------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def _leapfrog_regularize(data, data_old, grad, nu, sigma, exclude, q) -> np.float32_t:
    return __leapfrog_regularize(data, data_old, grad, nu, sigma, exclude, data.shape[0], q)

"""
The MIT License (MIT)

Copyright (c) 2016 International Audio Laboratories Erlangen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

""" Module for calculating DCT type 4 using FFT and pre/post-twiddling

.. warning::
    These core transforms will produce aliasing when used without overlap.
    Please use :py:mod:`mdct` unless you know what this means.

"""

#from __future__ import division
import numpy
import scipy

__all__ = [
    'mdct', 'imdct',
    'mdst', 'imdst',
    'cmdct', 'icmdct',
    'mclt', 'imclt',
]


def mdct(x, odd=True):
    """ Calculate modified discrete cosine transform of input signal

    Parameters
    ----------
    X : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.

    Returns
    -------
    out : array_like
        The output signal

    """
    return numpy.real(cmdct(x, odd=odd)) * numpy.sqrt(2)


def imdct(X, odd=True):
    """ Calculate inverse modified discrete cosine transform of input signal

    Parameters
    ----------
    X : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.

    Returns
    -------
    out : array_like
        The output signal

    """
    return icmdct(X, odd=odd) * numpy.sqrt(2)


def mdst(x, odd=True):
    """ Calculate modified discrete sine transform of input signal

    Parameters
    ----------
    X : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.

    Returns
    -------
    out : array_like
        The output signal

    """
    return -1 * numpy.imag(cmdct(x, odd=odd)) * numpy.sqrt(2)


def imdst(X, odd=True):
    """ Calculate inverse modified discrete sine transform of input signal

    Parameters
    ----------
    X : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.

    Returns
    -------
    out : array_like
        The output signal

    """
    return -1 * icmdct(X * 1j, odd=odd) * numpy.sqrt(2)


def cmdct(x, odd=True):
    """ Calculate complex MDCT/MCLT of input signal

    Parameters
    ----------
    x : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.

    Returns
    -------
    out : array_like
        The output signal

    """
    N = len(x) // 2
    n0 = (N + 1) / 2
    if odd:
        outlen = N
        pre_twiddle = numpy.exp(-1j * numpy.pi * numpy.arange(N * 2) / (N * 2))
        offset = 0.5
    else:
        outlen = N + 1
        pre_twiddle = 1.0
        offset = 0.0

    post_twiddle = numpy.exp(
        -1j * numpy.pi * n0 * (numpy.arange(outlen) + offset) / N
    )

    X = scipy.fftpack.fft(x * pre_twiddle)[:outlen]

    if not odd:
        X[0] *= numpy.sqrt(0.5)
        X[-1] *= numpy.sqrt(0.5)

    return X * post_twiddle * numpy.sqrt(1 / N)


def icmdct(X, odd=True):
    """ Calculate inverse complex MDCT/MCLT of input signal

    Parameters
    ----------
    X : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.

    Returns
    -------
    out : array_like
        The output signal

    """
    if not odd and len(X) % 2 == 0:
        raise ValueError(
            "Even inverse CMDCT requires an odd number "
            "of coefficients"
        )

    X = X.copy()

    if odd:
        N = len(X)
        n0 = (N + 1) / 2

        post_twiddle = numpy.exp(
            1j * numpy.pi * (numpy.arange(N * 2) + n0) / (N * 2)
        )

        Y = numpy.zeros(N * 2, dtype=X.dtype)
        Y[:N] = X
        Y[N:] = -1 * numpy.conj(X[::-1])
    else:
        N = len(X) - 1
        n0 = (N + 1) / 2

        post_twiddle = 1.0

        X[0] *= numpy.sqrt(2)
        X[-1] *= numpy.sqrt(2)

        Y = numpy.zeros(N * 2, dtype=X.dtype)
        Y[:N+1] = X
        Y[N+1:] = -1 * numpy.conj(X[-2:0:-1])

    pre_twiddle = numpy.exp(1j * numpy.pi * n0 * numpy.arange(N * 2) / N)

    y = scipy.fftpack.ifft(Y * pre_twiddle)

    return numpy.real(y * post_twiddle) * numpy.sqrt(N)

mclt = cmdct
imclt = icmdct
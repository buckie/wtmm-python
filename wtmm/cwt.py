from __future__ import division, absolute_import, print_function

__author__ = 'wjm'

from scipy import signal
from numpy import log
import numpy as np
import matplotlib.pyplot as plt

from wtmm.tracing import skeletor


def create_w_coef_mask(w_coefs, epsilon=0.1, order=1):
    """
    Create a new matrix, the same shape as the wavelet coefficient one, but with zeros everywhere except for local
    maxima's. Epsilon here is used for ranking the strength of the local maxima.

    Assumes that the coefficient matrix coming in is already in absolute terms

    :param w_coefs: wavelet coefficient matrix
    :param epsilon: divided against the maxima, used for transparent ranking
    :param order: how many neighboors on a given row to look at to determine maxima
    :return: same shape array, see above
    """
    mask = np.zeros_like(w_coefs, dtype=int)
    for n, row in enumerate(w_coefs):
        maxs = signal.argrelmax(row, order=order)[0]
        mask[n, maxs] = row[maxs] / epsilon

    return mask


def wtmm(sig, width_step=0.5, max_scale=None, wavelet=signal.ricker, epsilon=0.1,
         order=1, smallest_scale=0, proximity=9, corona_prox=1, top_threshold=0.05, plot=True):
    """
    Just a fast path to run perform_cwt and skeletor together

    :param sig: 1 dimensional array -- the signal to be hit with the wavelet
    :param width_step: what width step to use between the min and the max
    :param max_scale: the maximum scale to use. Default = len(sig)/4
    :param wavelet: what wavelet to use as the mother
    :param epsilon: how to score the maxima's intensity (e.g. intensity / epsilon )
    :param order: how many neighbors to look at when finding the local maxima
    :param smallest_scale: the smallest scale to look at in search of skeletons
    :param proximity: how close to look for the next step during skeleton construction
    :param plot: whether to plot the original CWT coefficient matrix as a heatmap
    :param corona_prox: proximity used to test for matched coronal loops
    :param top_threshold: percent distance from max-row to use for escaping cutoff

    :return: the mask, see above
    """
    w_coef = perform_cwt(sig,
                         width_step=width_step,
                         max_scale=max_scale,
                         wavelet=wavelet,
                         epsilon=epsilon,
                         order=order,
                         plot=plot)

    bifurcations = skeletor(w_coef, proximity=proximity, smallest_scale=smallest_scale,
                            plot=plot, corona_prox=corona_prox, top_threshold=top_threshold)

    return bifurcations


def perform_cwt(sig, width_step=0.5, max_scale=None, wavelet=signal.ricker, epsilon=0.1, order=1, plot=True):
    """
    Perform the continuous wavelet transform against the incoming signal. This function will normalize the signal
    (to 0-1 in the y axis) for you, as well as taking the -1 * abs( log( ) ) of the matrix that is found. Literature
    suggests that len/4 is a good balance for finding the bifurcations vs execution time

    This will automatically create the maxima only mask of the wavelet coef matrix for you. To see the original, use
    plot=True
    :param sig: 1 dimensional array -- the signal to be hit with the wavelet
    :param width_step: what width step to use between the min and the max
    :param max_scale: the maximum scale to use. Default = len(sig)/4
    :param wavelet: what wavelet to use as the mother
    :param epsilon: how to score the maxima's intensity (e.g. intensity / epsilon )
    :param order: how many neighbors to look at when finding the local maxima
    :param plot: whether to plot the original CWT coefficient matrix as a heatmap
    :return: the mask, see above
    """
    if not max_scale:
        max_scale = len(sig) / 4
    widths = np.arange(1, max_scale, width_step)

    # normalize the signal to fit in the wavelet
    sig_max = sig.max()
    sig_min = sig.min()
    sig = (sig - (sig_min - 0.01)) / (sig_max - sig_min + 0.02)


    # Run the transform
    w_coefs = abs(-1 * log(abs(signal.cwt(sig, wavelet, widths))))

    # Create the mask, keeping only the maxima
    mask = create_w_coef_mask(w_coefs, epsilon=epsilon, order=order)

    if plot:
        plt.figure(figsize=(14, 10))
        plt.pcolormesh(w_coefs)
        plt.colorbar()
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.xaxis.tick_top()
        plt.show()

    return mask

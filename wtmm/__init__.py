from __future__ import division, absolute_import, print_function

__author__ = 'wjm'
__all__ = [
    'wtmm',
    'walk_bifurcation',
    'perform_cwt',
    'skeletor',
    '_sort_tuples',
    '_possible_direction',
    '_create_w_coef_mask',
]

from collections import OrderedDict
from scipy import signal
from numpy import log

import numpy as np
import matplotlib.pyplot as plt


def _sort_tuples(x, y):
    """
    Special sort for the tuples, used to rank which point in the hood is "next". The preference is closest,
    then strongest
    :param x: tuple of (distance, strength)
    :param y: tuple of (distance, strength)
    :return:
    """
    if x[0] < y[0]:
        return -1
    elif x[0] == y[0]:
        if x[1] > y[1]:
            return -1
        elif x[1] < y[1]:
            return 1
        else:
            return 0
    else:
        return 1


def _possible_direction(hood, center_row=3, center_col=3):
    """
    For a given region of space in the wavelet coefficient matrix (the neighborhood) find the non-zero points,
    make them into a list, and sort them according to their net distance.

    :param hood: neighborhood to look in, a subset of the coef matrix
    :param center_row: location of the current point in the hood's rows
    :param center_col: location of the current point in the hood's columns
    :return: the best match (head of the list)
    """
    nzs = np.where(hood[:center_row] > 0)

    # tuple of (abs-dist, value, (row, col))
    matches = [(abs(row - center_row) + abs(col - center_col), hood[row, col], (row, col)) for row, col in zip(*nzs)]
    matches.sort(cmp=_sort_tuples)

    # turn this on for a sanity check
    # assert hood[center_row, center_col] != 0, matches
    return matches


def walk_bifurcation(mtx, start_row, start_col, proximity=9):
    """
    For a given wavelet coefficient matrix, derive a contiguous line for any given starting point. Starting point
    must be non-zero. This is used to find where the line "strikes" the ground, not to find where they start.


    :param mtx: wavelet coefficient matrix
    :param start_row: row of leading non-zero element
    :param start_col: column of the same
    :param proximity: how far this function should look in the vicinity of the current point. Default is 9
    :return: tuple(bool, list of coordinates) - bool for if the line hits the ground, coodinates of the points consumed
    :raise ValueError:
    """
    signal_strengths = []
    center_row, center_col = start_row, start_col
    max_row, max_col = [i - 1 for i in mtx.shape]
    trace_rt = []

    while center_row > 0:

        # get the proximity bounds for a given point in the matrix (addresses to look in)
        right_bound = center_col + proximity + 1
        left_bound = center_col - proximity
        hood_center_col = proximity
        if right_bound > max_col:
            right_bound = max_col
        elif left_bound < 0:
            # edge case when the hood extends beyond the bounds of the matrix
            # center in the hood is usually proximity, but if the left_bound is in conflict
            # then we have to adjust the center. As left_bound is negative at this point,
            # it is also the amount of shift the center needs...
            # eg:
            # proximity = 3
            # center_col = 2
            # left_bound = -1
            # hood[-1] = [0, 0, center, 0, 0 ,0] <-- hood[-1] is always hood_center_row
            # thus hood_center_col need to have -1 applied (or just the invalid left_bound)
            hood_center_col = proximity + left_bound
            left_bound = 0

        lower_bound = center_row - proximity
        if lower_bound < 0:
            # same argument as above applies
            hood_center_row = proximity + lower_bound
            lower_bound = 0
        else:
            hood_center_row = proximity


        # get the neighborhood of addresses...
        hood = mtx[lower_bound:center_row + 1, left_bound:right_bound]

        # find the best choice for the ridge
        possibles = _possible_direction(hood, center_row=hood_center_row, center_col=hood_center_col)

        if not possibles:
            # Means we've failed to hit the ground -- the algorithm will walk the path all the way to 0 or fail
            return False, trace_rt

        # get the winner
        match = possibles.pop(0)

        # recompute the center of the addresses and continue
        _, strength, (match_hood_row, match_hood_col) = match



        # match_hood_row < proximity always (this moves us up the matrix rows) but is always off by 1
        center_row += match_hood_row - hood_center_row
        # this can be +/- depending on the direction
        center_col += match_hood_col - hood_center_col
        # track the signal strength as well
        signal_strengths.append(strength)

        # TODO: we need to keep track of the movement of the curves

        if center_row >= 0:
            trace_rt.append((center_row, center_col))
        else:
            # Means that the next step takes us out of the matrix, so we are done
            trace_rt.append((0, center_col))

        if center_col == max_col or center_col == 0:
            # If we end up on and edge, this is not a valid bifurcation
            return False, trace_rt

    return True, trace_rt


def __print_hood(hood):
    """
    Utility function useful for debugging the path finder
    :param hood: neighborhood matrix
    """
    print(hood)
    plt.figure(figsize=(14, 10))
    plt.pcolormesh(hood, cmap='Greys')
    plt.colorbar()
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.xaxis.tick_top()
    plt.show()


def skeletor(mtx, proximity=9, smallest_scale=0, top_threshold=0.05, corona_prox=1, plot=False):
    '''
    Skeleton Constructor

    The basic ideas is to scan the coefficient matrix from max_row to 0 looking for non-zero elements. It assumes that
    the matrix has already been cleaned of everything that is not a local maxima. I generally use order=1 for this

    Warning: Has side-effects. To simplify the algorithm, any point consumed by this function will be zeroed out

    :param mtx: wavelet coefficient matrix
    :param proximity: how near by a non-zero point to look for the next non-zero point to jump to
    :param smallest_scale: where to draw the line in the search for bifurcations
    :param plot: plot the skeleton that is constructed
    proximity: defines region around the matrix
    '''
    # NB: scale <-> row
    # NB: shift <-> col
    max_row, max_col = mtx.shape
    max_row -= 1
    max_col -= 1

    # holder for the ridges
    bifurcations = OrderedDict()
    invalids = OrderedDict()
    bi_cnt = 0

    for n, row_num in enumerate(range(max_row, smallest_scale, -1)):
        # loop from the max-scale up
        maxs = signal.argrelmax(mtx[row_num])[0]
        # print "loop", n, "row_num", row_num, "maxs", maxs

        if not maxs.any():
            # Nothing here...
            # print "exit", maxs.any()
            continue

        for start_pt in maxs:
            continuous, bifurc_path = walk_bifurcation(mtx, row_num, start_pt, proximity=proximity)

            if continuous:
                # add the bifurcation to the collector; key == row[0] intercept's column number
                bifurcations[(bi_cnt, bifurc_path[-1][1])] = bifurc_path
                bi_cnt += 1
            elif bifurc_path:
                invalids[bifurc_path[-1]] = bifurc_path

            if len(bifurc_path):
                # now zero out all of the entries that were used to walk the bifucation
                rows_b, cols_b = zip(*bifurc_path)
                rows_b = np.array(rows_b)
                cols_b = np.array(cols_b)

                mtx[rows_b, cols_b] = 0

    coronae = match_coronae(bifurcations,
                            top_threshold=int(max_row-(top_threshold*max_row)),
                            neighborhood_threshold=corona_prox)

    if plot:
        plt.figure(figsize=(14, 10))
        for n, (k, v) in enumerate(coronae.items()):
            rows, cols = zip(*v)
            plt.plot(cols, rows)
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.xaxis.tick_top()
        plt.show()

    return coronae


def _create_w_coef_mask(w_coefs, epsilon=0.1, order=1):
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
    mask = _create_w_coef_mask(w_coefs, epsilon=epsilon, order=order)

    if plot:
        plt.figure(figsize=(14, 10))
        plt.pcolormesh(w_coefs)
        plt.colorbar()
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.xaxis.tick_top()
        plt.show()

    return mask


def wtmm(sig, width_step=0.5, max_scale=None, wavelet=signal.ricker, epsilon=0.1,
         order=1, smallest_scale=0, proximity=9, plot=True):
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
    :return: the mask, see above
    """
    w_coef = perform_cwt(sig,
                         width_step=width_step,
                         max_scale=max_scale,
                         wavelet=wavelet,
                         epsilon=epsilon,
                         order=order,
                         plot=plot)

    bifurcations = skeletor(w_coef, proximity=proximity, smallest_scale=smallest_scale, plot=plot)

    return bifurcations



def get_distance(rc0, rc1):
    '''
    Get the distance between two row-column points
    '''
    r0, c0 = rc0
    r1, c1 = rc1
    return ( (c1 - c0)**2 + (r1 - r0)**2 )**0.5


def get_best_match(matched_lines, cur_pts, bifucs, neighborhood_threshold=1):

    # lines in the hood
    matches = []

    threshold = (cur_pts[0][0]**0.5)*(neighborhood_threshold)//1
    if threshold < 3:
        threshold = 3

    # after handling a full traversal of the row domain, we can be sure that the corona are self contianed within the graph
    for (n,v), pts in bifucs.iteritems():

        # Again, guard against a double match with the added benefit of speeding up the loop.
        # Things that have already been matched are also impossible to match against but would usually be processed first
        # so skip them.
        if n in matched_lines:
            continue

        # grab the top most point as our anchor -- what we match against.
        anchor = pts[0]

        # Now we walk pts and check each against the distance to the anchor. We want to track what the current minimum distance
        # because once it begins to increase, we can stop the algorithm. Further, we can default the best_distance value
        # to the distance between the anchor and the start of pts.

        best_distance = get_distance(cur_pts[0], anchor)
        pts_idx = None

        for idx, pt in enumerate(cur_pts):
            dist = get_distance(pt, anchor)
            if dist < best_distance:
                best_distance = dist
                pts_idx = idx
            elif dist > best_distance:
                break

        # We only want to track distances that better the last iteration's distance
        if best_distance <= threshold:
            matches.append(((n, v), pts_idx, best_distance))
            break

    if matches:
        # Just defaulting here to avoid a branching if-else statement
        winning_match = matches[0]
        for match in matches:
            # test the distance of each element of the list... easier than making a custom sort
            if match[2] < winning_match[2]:
                winning_match = match
        return winning_match
    else:
        return (None, None, None)

def match_coronae(bifucs, top_threshold, neighborhood_threshold=1):
    '''
    Takes the individual bifurcation lines and returns an ordered dict of (rank, x-axis point, corona -> bool): [(row,col)]
    '''

    matched_lines = set()
    coronae = OrderedDict()
    for i, ((n,v), pts) in enumerate(bifucs.iteritems()):

        # check that the current item hasn't already been matched against
        if n in matched_lines:
            continue
        else:
            # if the current number is not matched, then add it now
            matched_lines.add(n)

        # get the starting coordinates
        row_max, col_max = pts[0]
        row_min, col_min = pts[-1]

        # handle lines that traverse nearly the entire graph differently
        if row_max > top_threshold:
            coronae[(i, v, (pts[-1][1]))] = pts
            matched_lines.add(n)
            continue

        # This returns the first match against the current segment. The first should be the best.
        # pts_idx is the place to snip the tail off of the current line and glue the match and the current line together
        match_key, pts_idx, _ = get_best_match(matched_lines, pts, bifucs)

        if not match_key:
            coronae[(i, v, (pts[0][1]))] = pts
            continue
        match_pts = bifucs[match_key]

        # left and right here are just for clarity in logic, the parts may in fact be reversed in order (but it doesn't matter)
        left = pts[pts_idx:]
        right = match_pts

        # the 0-index has the max row for each part
        corona_max = left[0][0] if left[0][0] > right[0][0] else right[0][0]
        # now glue it all together
        corona = left[::-1] + right[:]

        coronae[(i, v, (corona[0][1],corona[-1][0]))] = corona

        # add the components to the matched set
        matched_lines.add(n)
        matched_lines.add(match_key[0])

    return coronae

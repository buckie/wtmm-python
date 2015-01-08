__author__ = 'wjm'

import numpy as np
import matplotlib.pyplot as plt


def financial_cartoon(Iterations=10, Multifractal=1, noise_type=False, noise_level=1.0, plot=False):
    if Multifractal:
        turns = ((0.25, 0.5), (0.75, 0.25))
    else:
        turns = ((0.4, 0.6), (0.6, 0.4))
    first_turn, second_turn = turns
    ys = [0, 1]
    ts = [0, 1]

    if not noise_type:
        for i in range(0, Iterations + 1):

            j = 0
            while ts[j] < 1:
                dt = ts[j + 1] - ts[j]
                dy = ys[j + 1] - ys[j]

                ts.insert(j + 1, ts[j] + first_turn[0] * dt)
                ts.insert(j + 2, ts[j] + second_turn[0] * dt)
                ys.insert(j + 1, ys[j] + first_turn[1] * dy)
                ys.insert(j + 2, ys[j] + second_turn[1] * dy)

                j += 3
    else:
        if noise_type == 'uniform':
            noise = np.random.rand
        elif noise_type == 'normal':
            noise = np.random.randn
        else:
            raise ValueError('Only normal and uniform accepted for noise at this time')

        for i in range(0, Iterations + 1):

            j = 0
            while ts[j] < 1:
                dt = ts[j + 1] - ts[j]
                dy = ys[j + 1] - ys[j]

                # normalize the noise versus the current dt
                n_a, n_b = (noise(2) * noise_level) * float(dy)

                ts.insert(j + 1, ts[j] + first_turn[0] * dt)
                ts.insert(j + 2, ts[j] + second_turn[0] * dt)
                ys.insert(j + 1, ys[j] + n_a + first_turn[1] * dy)
                ys.insert(j + 2, ys[j] + n_b + second_turn[1] * dy)

                j += 3

    if plot:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'axisbg': '#EEEEEE', 'axisbelow': True})
        ax.grid(color='w', linewidth=2, linestyle='solid')
        ax.plot(ts, ys, color='b', alpha=0.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    return np.array(ts), np.array(ys)
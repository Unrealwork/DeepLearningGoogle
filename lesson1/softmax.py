"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np
import math


def softmax(scores_array):
    sum = 0
    exp_arr = map(np.exp, scores_array)
    for num in exp_arr:
        sum += num
    return np.asarray(map(lambda x: x / sum, exp_arr))


print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt

x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()

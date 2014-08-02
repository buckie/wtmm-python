wtmm-python
===========

Implementation of the Wavelet Transform Modulus Maxima algorithm in python for the purpose of deconstructing 1-D time series data into a peicewise linear representation, parititioned according to similarity.

## Research

### Folder 

Collection of relevant papers.

## Results

After much work, the method seems to be stable and working. More refinement needs to be done, but that can wait for later. See [this notebook](http://nbviewer.ipython.org/github/buckie/wtmm-python/blob/master/notebooks/scipy%20approach.ipynb) for more details.

As a preview, here is a demo of the results:

* black line: original signal (fractional Brownian motion)
* yellow line: piecewise vector decomposition, similar to a "backed out" financial cartoon.
* subsequent lines: partitioned segments of the original signal, based on local Holder exponent extrema. More extrema are used for the partitioning as you descend.

![Demo of Partitioning](https://github.com/buckie/wtmm-python/raw/master/clustering_demo.png)


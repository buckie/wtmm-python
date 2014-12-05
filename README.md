wtmm-python
===========

Implementation of the Wavelet Transform Modulus Maxima algorithm in python for the purpose of deconstructing 1-D time series data into a piecewise linear representation, partitioned according to similarity.

### Replicate Environment

This uses python2 and the easiest way to install the environment is to install Continuum's Anaconda and run the following command:

    conda create -n wtmm python=2.7 scipy numpy matplotlib ipython-notebook pip setuptools pandas
    source activate wtmm

## Research

### Folder 

Collection of relevant papers.

## Results

After much work, the method seems to be stable and working. More refinement needs to be done, but that can wait for later. 

To view all the results (tested against brownian motion, fractional brownian motion, sin wave + each of the following w/ uniform random or normal random jitters added) navigate to the parent folder in the notebook viewer: [link](http://nbviewer.ipython.org/github/buckie/wtmm-python/tree/master/notebooks/)

As a preview, here is a demo of the results -- see [this notebook](http://nbviewer.ipython.org/github/buckie/wtmm-python/blob/master/notebooks/scipy%20approach.ipynb) for more details:

**black line**: original signal (fractional Brownian motion)

**yellow line**: piecewise vector decomposition, similar to a "backed out" financial cartoon.

**subsequent lines**: partitioned segments of the original signal, based on local Holder exponent extrema. More extrema are used for the partitioning as you descend.

![Demo of Partitioning](https://github.com/buckie/wtmm-python/raw/master/notebooks/clustering_demo.png)


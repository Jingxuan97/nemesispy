# -*- coding: utf-8 -*-
"""
Plot posterior distribution with corner plots.
Script to visualise retrieval outcome.
"""

# from turtle import title
import matplotlib
matplotlib.use('PDF')
import numpy as np
import matplotlib.pyplot as plt
import pymultinest
import pickle
import os
import sys
from corner import corner
# from wasp43b import NemesisAPI, static_param

print('module loaded')
"""Just vmr
### Retrieved parameter names and truth values
labels = np.array([
    'log$_{10}$ H$_2$O',
    'log$_{10}$ CO$_2$',
    'log$_{10}$ CO',
    'log$_{10}$ CH$_4$',
])
truths = np.array([
    -3.3190755511940075,
    -7.131446073540052,
    -3.333162032372402,
    -6.877021089673744
    ])
ranges = np.array([
    (-8,-2),
    (-8,-2),
    (-8,-2),
    (-8,-2),
])
n_params = 8


### Get retrieval info from the chains folder
#prefix = './out'
analyzer = pymultinest.Analyzer(
    n_params=n_params,
    outputfiles_basename='chains3/fix-'
    )
values = analyzer.get_equal_weighted_posterior()
params = values[:, (n_params-4):n_params]
lnprob = values[:, -1]
samples = params
Nsamp = values.shape[0]
Npars = n_params

index, cube = np.loadtxt('chains3/fix-stats.dat',skiprows=26,unpack=True)
MAPs = cube[-4:]
print('MAPs',MAPs)
### Plot Samples: Triangle Plot
# matplotlib.rcParams.update({'font.size': 18})
print('Nsamp\n', Nsamp, 'Npars\n', Npars)
print('sample\n', values)
print('Nsamp\n', values.shape)
figure=corner(
    xs = samples,
    bins = 20,
    color = 'navy',
    # smooth =  0.5,
    # smooth1d = 0.5,
    smooth = None,
    labels = labels,
    label_kwargs = {'fontsize' : 'x-large'},
    show_titles = True,
    range = ranges,
    truths = truths,
    truth_color = 'black',
    MAPs = MAPs,
    MAP_color = 'tomato',
    quantiles = [0.16,0.5,0.84],
    plot_contours = True,
    scale_hist = False,
    )
figure.savefig("triangle_bins_100.pdf",
    dpi=1000)
plt.close()
"""

### Retrieved parameter names and truth values
labels = np.array([
    'log$_{10}$ H$_2$O',
    'log$_{10}$ CO$_2$',
    'log$_{10}$ CO',
    'log$_{10}$ CH$_4$',
])
truths = np.array([
    -3.3190755511940075,
    -7.131446073540052,
    -3.333162032372402,
    -6.877021089673744
    ])
ranges = np.array([
    (-8,-2),
    (-8,-2),
    (-8,-2),
    (-8,-2),
])
n_params = 8


### Get retrieval info from the chains folder
#prefix = './out'
analyzer = pymultinest.Analyzer(
    n_params=n_params,
    outputfiles_basename='chains3/fix-'
    )
values = analyzer.get_equal_weighted_posterior()
params = values[:, (n_params-4):n_params]
lnprob = values[:, -1]
samples = params
Nsamp = values.shape[0]
Npars = n_params

index, cube = np.loadtxt('chains3/fix-stats.dat',skiprows=26,unpack=True)
MAPs = cube[-4:]
print('MAPs',MAPs)
### Plot Samples: Triangle Plot
# matplotlib.rcParams.update({'font.size': 18})
print('Nsamp\n', Nsamp, 'Npars\n', Npars)
print('sample\n', values)
print('Nsamp\n', values.shape)
figure=corner(
    xs = samples,
    bins = 20,
    color = 'navy',
    # smooth =  0.5,
    # smooth1d = 0.5,
    smooth = None,
    labels = labels,
    label_kwargs = {'fontsize' : 'x-large'},
    show_titles = True,
    range = ranges,
    truths = truths,
    truth_color = 'black',
    MAPs = MAPs,
    MAP_color = 'tomato',
    quantiles = [0.16,0.5,0.84],
    plot_contours = True,
    scale_hist = False,
    )
figure.savefig("triangle_bins_100.pdf",
    dpi=1000)
plt.close()

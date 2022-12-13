import matplotlib
matplotlib.use('PDF')
import numpy as np
import matplotlib.pyplot as plt
import pymultinest
import pymultinest
import pickle
import os
import sys
from nemesispy.retrieval.corner import corner

n_params = 24
a = pymultinest.Analyzer(n_params=n_params,
    outputfiles_basename='chains2/full-')
values = a.get_equal_weighted_posterior()

params = values[:, :n_params]
lnprob = values[:, -1]
samples = params
Nsamp = values.shape[0]
figure=corner(xs = samples,
              plot_contours = 'True',
              smooth =  0.5,
              smooth1d = 0.5,
              bins = 20,
              color = 'b',
              quantiles = [0.16,0.5,0.84],
              # labels = titles,
              # range = [(-10,-1),(-10,-1),(-10,-1),(-10,-1),(-4,1),(-4,1),(-4,1),(0,1),(0,3000)],
              show_titles = 'True',
              # truths = truths,
              scale_hist = False)

figure.savefig("triangle1.pdf")
plt.close()

params = values[:, -5:n_params]
lnprob = values[:, -1]
samples = params
Nsamp = values.shape[0]
figure=corner(xs = samples,
              plot_contours = 'True',
              smooth =  0.5,
              smooth1d = 0.5,
              bins = 20,
              color = 'b',
              quantiles = [0.16,0.5,0.84],
              # labels = titles,
              # range = [(-10,-1),(-10,-1),(-10,-1),(-10,-1),(-4,1),(-4,1),(-4,1),(0,1),(0,3000)],
              show_titles = 'True',
              # truths = truths,
              scale_hist = False)

figure.savefig("triangle2.pdf")
plt.close()
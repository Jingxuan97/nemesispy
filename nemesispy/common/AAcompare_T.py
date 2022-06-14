# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
gcm = np.loadtxt('AAgcm_reshaped.txt',ndmin=2,delimiter=',')
best_fit = np.loadtxt('AAbest_fit.txt',ndmin=2,delimiter=',')

diff = gcm-best_fit
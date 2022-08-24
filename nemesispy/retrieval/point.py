#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Test retrieval of a single emmission spectrum.
"""
import numpy as np
import os
from nemesispy.radtran.forward_model import ForwardModel
ktable_path = "/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables"
cia_file_path = "/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/cia"\
    +'exocia_hitran12_200-3800K.tab'
lowres_file_paths = [
    'h2owasp43.kta',
    'co2wasp43.kta',
    'cowasp43.kta',
    'ch4wasp43.kta']
for ipath,path in enumerate(lowres_file_paths):
    lowres_file_paths[ipath] = os.path.join(ktable_path,path)

# Forward Model


# Create a single emmision model using Guillot
P_grid = np.geomspace(10*1e5,1e-3*1e5,20)
g_plt = 25 # m s-2
T_eq = 1200 # K
k_IR = 1e-1 # m2 kg-1
gamma = 1e-1
f = 1
TP = np.array([2520.55401599, 2518.75290023, 2517.64176093, 2516.9567351 ,
       2516.53458472, 2516.27449842, 2516.1142846 , 2516.01560193,
       2515.95449081, 2515.83560878, 2513.46120885, 2496.0300132 ,
       2441.65321758, 2341.85613232, 2208.68970501, 2062.55178614,
       1921.53209157, 1797.65457841, 1696.81747349, 1619.94123111])

# Create a single emission spectrum
wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875,
       1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6   ,
       4.5   ])
FM = ForwardModel()
FM.set_opacity_data(kta_file_paths=ktable_path,cia_file_path=cia_file_path)
FM.calc_point_spectrum_hydro()

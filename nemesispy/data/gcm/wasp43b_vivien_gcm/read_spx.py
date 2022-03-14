# -*- coding: utf-8 -*-
import numpy as np
wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875,
       1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6   ,
       4.5   ])

phase_grid = np.array([ 22.5,  45. ,  67.5,  90. , 112.5, 135. , 157.5, 180. , 202.5,
       225. , 247.5, 270. , 292.5, 315. , 337.5])

NPHASE = len(phase_grid)
NWAVE = len(wave_grid)

phase_curve = np.zeros((NPHASE,NWAVE,2)) # phase x spec

phase_plot = np.zeros((NWAVE,NPHASE,2))

iline = 0


for iphase,phase in enumerate(phase_grid):
    heading = np.loadtxt('wasp43b_phase_curve.txt',skiprows=iline,max_rows=1)
    iline += 1
    nwave = int(np.loadtxt('wasp43b_phase_curve.txt',skiprows=iline,max_rows=1))
    iline += 1
    nfov = int(np.loadtxt('wasp43b_phase_curve.txt',skiprows=iline,max_rows=1))
    iline += 1
    fov_data = np.loadtxt('wasp43b_phase_curve.txt',skiprows=iline,max_rows=nfov)
    iline += nfov
    wave,spec,error = np.loadtxt('wasp43b_phase_curve.txt',unpack=True,
                                 skiprows=iline,max_rows=nwave)
    iline += nwave
    
    for iwave in range(nwave):
        phase_curve[iphase,iwave,0] =  spec[iwave]
        phase_curve[iphase,iwave,1] =  error[iwave]
    
    for iwave in range(nwave):
        phase_plot[iwave,iphase,0] = spec[iwave]
        phase_plot[iwave,iphase,1] = error[iwave]
        
        
    


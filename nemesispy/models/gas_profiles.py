#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def gen_vmrmap_1d(h2o,co2,co,ch4,npress,h2_frac = 0.84):
    """
    Generate a 1D uniform gas abundance map.
    The abundance map is defined on a pressure grid.

    Parameters
    ---------

    Returns
    -------
    vmr_grid
    """
    he_frac = 1 - h2_frac
    vmr_grid = np.ones((npress,6))
    vmr_grid[:,0] *= 10**h2o
    vmr_grid[:,1] *= 10**co2
    vmr_grid[:,2] *= 10**co
    vmr_grid[:,3] *= 10**ch4
    vmr_grid[:,4] *= he_frac * (1-10**h2o-10**co2-10**co-10**ch4)
    vmr_grid[:,5] *= h2_frac * (1-10**h2o-10**co2-10**co-10**ch4)
    return vmr_grid

def gen_vmrmap1(h2o,co2,co,ch4,nlon,nlat,npress,
        h2_frac = 0.84):
    """
    Generate a 3D uniform gas abundance map.
    The abundance map is defined on a (longitude,latitude,pressure) grid.

    Parameters
    ---------

    Returns
    -------
    vmr_grid
    """
    he_frac = 1 - h2_frac
    vmr_grid = np.ones((nlon,nlat,npress,6))
    vmr_grid[:,:,:,0] *= 10**h2o
    vmr_grid[:,:,:,1] *= 10**co2
    vmr_grid[:,:,:,2] *= 10**co
    vmr_grid[:,:,:,3] *= 10**ch4
    vmr_grid[:,:,:,4] *= he_frac * (1-10**h2o-10**co2-10**co-10**ch4)
    vmr_grid[:,:,:,5] *= h2_frac * (1-10**h2o-10**co2-10**co-10**ch4)
    return vmr_grid
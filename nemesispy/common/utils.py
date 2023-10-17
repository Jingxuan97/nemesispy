#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import numpy as np

def mkdir(folder_name):
    """
    Make a folder at path 'folder_name' if the folder path does not exist yet.

    Parameters
    ----------
    folder_name : str
        The path + name of the folder.

    Returns
    -------
    None
    """
    try:
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
    except FileExistsError:
        pass

def divide_gcm_grid(nlon,nlat,nrank):
    """
    Divide gcm grid points for multiple processors to be analysed in parallel.

    Parameters
    ----------
    nlon : int
        Number of longitudes in the general circulation model grid.
    nlat : int
        Number of latitudes in the general circulation model grid.
    nrank : int
        Number of processors to be used.

    Returns
    -------
    partition : ndarray
        A array containing nrank subarrays, where each subarray contain a number
        of [longitude_index, latitude_index] pairs.
    """
    njobs = (nlon) * (nlat)
    jobs = np.zeros((njobs,2))
    for ilon in range(nlon):
        for ilat in range(nlat):
            jobs[ilon*nlat+ilat,0] = ilon
            jobs[ilon*nlat+ilat,1] = ilat
    partition = np.array_split(jobs,nrank)
    return partition
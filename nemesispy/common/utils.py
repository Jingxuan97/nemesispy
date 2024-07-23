#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import numpy as np

def mkdir(folder_name):
    """
    Make a folder at 'folder_name' if the folder does not exist yet.

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

def bin_down(input_value, input_grid, bin_edges):
    """Bin down the input_value based on the input_grid and bin_edges.

    Args:
        input_value : ndarray
            An array of values to be binned down.
        input_grid : ndarray
            The grid on which the input_value is defined.
        bin_edges : ndarray
            Edges of the bins to be used for binning down.

    Returns:
        output_value : ndarray
            An array of binned down values.
    """
    output_value = np.zeros(len(bin_edges)-1)
    for iedge in range(len(bin_edges)-1):
        temp = []
        for index,value in enumerate(input_grid):
            if bin_edges[iedge]<value<bin_edges[iedge+1]:
                temp.append(input_value[index])
        output_value[iedge] = np.average(temp)
    return output_value
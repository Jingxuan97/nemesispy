import numpy as np
import matplotlib.pyplot as plt
wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875,
       1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6   ,
       4.5   ])

wavelengths,radiance = np.loadtxt('wasp43_stellar_newgrav.txt',skiprows=3,unpack=True)

interped_spec = np.interp(wave_grid,wavelengths,radiance)

wasp43_spec = np.array([3.341320e+25, 3.215455e+25, 3.101460e+25, 2.987110e+25,
       2.843440e+25, 2.738320e+25, 2.679875e+25, 2.598525e+25,
       2.505735e+25, 2.452230e+25, 2.391140e+25, 2.345905e+25,
       2.283720e+25, 2.203690e+25, 2.136015e+25, 1.234010e+24,
       4.422200e+23])


def interpolate_stellar_spectrum(wave_grid,wave_data,spec_data):

    assert np.all(np.diff(wave_data) > 0), "Need increaseing wave_dat"
    interped_spec = np.interp(wave_grid,wave_data,spec_data)

    return interped_spec
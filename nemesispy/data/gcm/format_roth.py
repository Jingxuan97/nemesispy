#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

roth_lon_grid = np.array(
    [-177.19, -171.56, -165.94, -160.31, -154.69, -149.06, -143.44,
     -137.81, -132.19, -126.56, -120.94, -115.31, -109.69, -104.06,
     -98.438, -92.812, -87.188, -81.562, -75.938, -70.312, -64.688,
     -59.062, -53.438, -47.812, -42.188, -36.562, -30.938, -25.312,
     -19.688, -14.062, -8.4375, -2.8125, 2.8125, 8.4375, 14.062,
     19.688, 25.312, 30.938, 36.562, 42.188, 47.812, 53.438,
     59.062, 64.688, 70.312, 75.938, 81.562, 87.188, 92.812,
     98.438, 104.06, 109.69, 115.31, 120.94, 126.56, 132.19,
     137.81, 143.44, 149.06, 154.69, 160.31, 165.94, 171.56, 177.19]
)

roth_lat_grid = np.array(
    [-87.188, -81.562, -75.938, -70.312, -64.688, -59.062, -53.438,
     -47.812, -42.188, -36.562, -30.938, -25.312, -19.688, -14.062,
     -8.4375, -2.8125, 2.8125, 8.4375, 14.062, 19.688, 25.312,
     30.938, 36.562, 42.188, 47.812, 53.438, 59.062, 64.688,
     70.312, 75.938, 81.562, 87.188]
)

roth_p_grid = np.array(
    [170.64, 120.54, 85.152, 60.152, 42.492, 30.017, 21.204, 14.979,
     10.581, 7.4747, 5.2802, 3.73, 2.6349, 1.8613, 1.3148, 0.92882,
     0.65613, 0.4635, 0.32742, 0.23129, 0.16339, 0.11542, 0.081532,
     0.057595, 0.040686, 0.028741, 0.020303, 0.014342, 0.010131,
     0.0071569, 0.0050557, 0.0035714, 0.0025229, 0.0017822, 0.0012589,
     0.00088933, 0.00062823, 0.00044379, 0.0003135, 0.00022146, 0.00015644,
     0.00011051, 7.8066e-05, 5.5146e-05, 3.8956e-05, 2.7519e-05, 1.944e-05,
     1.3732e-05, 9.7006e-06, 6.8526e-06, 4.8408e-06, 3.4196e-06, 2.4156e-06]
)

nlon_grid = len(roth_lon_grid)
nlat_grid = len(roth_lat_grid)
np_grid = len(roth_p_grid)

def gen_file_name(Teq,LogMet,LogG,Mstar,TiOVO='false'):
    Teq = int(Teq)
    if Teq not in [1000,1200,1400,1600,1800,2000,2200,2400]:
        raise Exception('Teq not in the grid')
    if LogMet not in [0.0,0.7,1.5]:
        raise Exception('Log metallicity not in the grid')
    if LogG not in [0.8,1.3,1.8]:
        raise Exception('LogG not in the grid')
    if Mstar not in [0.8,1.1,1.5]:
        raise Exception('Mstar not in the grid ')
    name = 'PTprofiles_halph_res-Teq_'
    name += str(Teq) +'-LogMet_'+str(LogMet)+'-LogDrag_0-Mstar_'+str(Mstar)\
        +'-Rp_1.3-logG_'+str(LogG)+'-TiOVO_'+TiOVO+'.dat'
    return name

def read_gcm_PT_roth(input_file_path):
    """
    Read a GCM temperature map
    """
    # read data from file
    index,longitude,latitude,pressure,T \
        = np.genfromtxt(input_file_path,unpack=True,delimiter=',')
    # load the temperature map to a n_longitude x n_latitude x n_pressure grid
    tmap = np.zeros((nlon_grid,nlat_grid,np_grid))
    for ilon,lon in enumerate(roth_lon_grid):
        for ilat,lat in enumerate(roth_lat_grid):
            for ip, p in enumerate(roth_p_grid):
                tmap[ilon,ilat,ip] = T[ilon*nlat_grid*np_grid+ilat*np_grid+ip]
                # check latitudes and longitudes match up
                assert (
                    (longitude[ilon*nlat_grid*np_grid+ilat*np_grid+ip]-lon)==0.0
                    ), 'error in reading gcm data longitudes'
                assert (
                    (latitude[ilon*nlat_grid*np_grid+ilat*np_grid+ip]-lat)==0.0
                    ), 'error in reading gcm data latitudes'

    # pressure in data file is increasing, so reverse pressure order
    for ilon,lon in enumerate(roth_lon_grid):
        for ilat,lat in enumerate(roth_lat_grid):
            tmap[ilon,ilat,:] = tmap[ilon,ilat,::-1]

    return tmap

def read_mean_params_roth(input_folder_name,nparams=4,skiprows=4,
        output_file_name='mean_params.txt'):
    """
    Read the posterior mean of 1D TP profile fit to the GCM at all grid points
    and save the output to a text file.

    Returns
    -------
    None
    """
    mean_parameters = np.zeros((nlon_grid * nlat_grid,nparams))
    for ilon in np.arange(nlon_grid):
        for ilat in np.arange(nlat_grid):
            index,means,sigmas\
                = np.loadtxt(
                    '{}/{}_{}-stats.dat'.format(input_folder_name,ilon,ilat),
                    skiprows=skiprows,
                    unpack=True,
                    max_rows=nparams
                    )
            mean_parameters[ilon*nlat_grid+ilat,:] = means
    np.savetxt('{}'.format(output_file_name),mean_parameters,delimiter=',')

def read_best_params_roth(input_folder_name,nparams=4,skiprows=11,
        output_file_name='best_params.txt'):
    """
    Read the MAP of 1D TP profile fit to the GCM at all grid points
    and save the output to a text file.

    Returns
    -------
    None
    """
    best_parameters = np.zeros((nlon_grid * nlat_grid,nparams))
    for ilon in np.arange(nlon_grid):
        for ilat in np.arange(nlat_grid):
            index,MAP\
                = np.loadtxt(
                    '{}/{}_{}-stats.dat'.format(input_folder_name,ilon,ilat),
                    skiprows=skiprows,
                    unpack=True,
                    max_rows=nparams
                    )
            best_parameters[ilon*nlat_grid+ilat,:] = MAP
    np.savetxt('{}'.format(output_file_name),best_parameters,delimiter=',')

def plot_param_contour_2stream(ouput_file_name,
    input_file_name = 'best_params.txt',
    param_name = [r'log $\gamma$',r'log $\kappa$',r'$f$',r'T$_{int}$'],
    dpi=400):
    """
    Plot the distribution of fit parameters.
    """
    nparam = len(param_name)
    params = np.loadtxt('{}'.format(input_file_name),
        unpack=True,delimiter=',')
    params = params.T

    ## Assign the parameters to their (longitude,latitude) grid positions
    best_params = np.zeros((nlon_grid,nlat_grid,nparam))
    for ilon in range(nlon_grid):
        for ilat in range(nlat_grid):
            best_params[ilon,ilat,:] = params[ilon*nlat_grid+ilat,:]

    ## Plot the best fitting 1D parameters
    # set up foreshortened latitude coordinates
    fs = np.sin(roth_lat_grid/180*np.pi)*90
    x,y = np.meshgrid(roth_lon_grid,fs,indexing='ij')
    xticks = np.array([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120,
        150, 180])
    # move the y ticks to the foreshortened location
    yticks_loc = np.sin(np.array([-60, -30, 0, 30, 60])/180*np.pi)*90
    yticks_label = np.array([-60, -30, 0, 30, 60])

    ## Set up multiplot
    fig,axs = plt.subplots(
        nrows=5,ncols=1,
        sharex=True,sharey=True,
        figsize=(8.3,11.7),
        dpi=400
    )

    for iparam,name in enumerate(param_name):
        # contour plot
        z_param = best_params[:,:,iparam]
        if iparam==2:
            z_param = 10**z_param
        im = axs[iparam].contourf(x,y,z_param,
                levels=20,
                cmap='magma',
                vmin=z_param.min(),
                vmax=z_param.max()
                )
        cbar = fig.colorbar(im,ax=axs[iparam])
        # axis setting
        axs[iparam].set_xticks(xticks)
        # axs[iparam].set_yticks(
        #     ticks=yticks_loc,labels=yticks_label)
        axs[iparam].set_title('{}'.format(name),#fontsize='small'
        )
    plt.yticks(yticks_loc,yticks_label)

    axs[4].axis('off')
    fig.tight_layout()
    plt.savefig('{}'.format(ouput_file_name),dpi=400)
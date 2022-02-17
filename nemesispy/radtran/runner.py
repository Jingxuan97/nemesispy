import sys
sys.path.append('/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/')
import matplotlib.pyplot as plt
import numpy as np
from nemesispy.data.constants import R_SUN, R_JUP_E, AMU, AU, M_JUP, R_JUP
from nemesispy.radtran.utils import calc_mmw
from nemesispy.radtran.models import Model2
from nemesispy.radtran.path import calc_layer # average
from nemesispy.radtran.read import read_kls
from nemesispy.radtran.radiance import radtran, calc_planck
from nemesispy.radtran.read import read_cia


class runner:

    def __init__(self, gas_file, cia_file):
        self.gas_file = gas_file
        self.cia_file = cia_file

from radtran.trig import gauss_lobatto_weights

### LIST OF STEPS TO COMPLETE A DISC AVERAGED SPECTRUM
# NEED static atm input: ID, ISO,
# NEED GLOBAL atm MODEL PROFILE INPUT: GLOBAL=VMR, GLOBAL-T, GLOBAL-H, GLOBL-P
# NEED static spectral input: wave_grid, g_ord, del_g, k_gas_w_g_p_t, Pgrid, Tgrid
# GLOBALVMR
"""
    0.0000     18.4349    161.5651     63.4349      0.0000  0.07701087
   35.1970     11.8202    143.1155     63.4349     49.3939  0.15402173
   61.8232    333.7203    115.0488     63.4349     50.2848  0.15402173
   50.5081    269.6843     89.7992     63.4349     37.5965  0.19146977
    0.0000    251.5651     71.5651     63.4349      0.0000  0.11445891
    0.0000    315.0000    135.0000      0.0000    180.0000  0.30901699
"""

fov_lat_long1 = np.array([[0.0000, 18.4349],
                [35.1970, 11.8202],
                [61.8232, 333.7203],
                [50.5081, 269.6843],
                [0.0000, 251.5651],
                [0.0000, 315.0000]])

lat_coord = np.linspace(0,90,num=10)
lon_coord = np.linspace(0,360,num=5)
nlat = len(lat_coord)
nlon = len(lon_coord )

lat_ax, lon_ax = np.meshgrid(lat_coord, lon_coord, indexing='ij')
nloc = len(lat_coord)*len(lon_coord)
global_model_lat_lon = np.zeros([nloc,2])

iloc=0
for ilat in range(nlat):
    for ilon in range(nlon):
        global_model_lat_lon[iloc,0] = lat_ax[ilat,ilon]
        global_model_lat_lon[iloc,1] = lon_ax[ilat,ilon]
        iloc+=1

global_model_lat_lon = np.array([[  0.,   0.],
                                [  0.,  90.],
                                [  0., 180.],
                                [  0., 270.],
                                [  0., 360.],
                                [ 10.,   0.],
                                [ 10.,  90.],
                                [ 10., 180.],
                                [ 10., 270.],
                                [ 10., 360.],
                                [ 20.,   0.],
                                [ 20.,  90.],
                                [ 20., 180.],
                                [ 20., 270.],
                                [ 20., 360.],
                                [ 30.,   0.],
                                [ 30.,  90.],
                                [ 30., 180.],
                                [ 30., 270.],
                                [ 30., 360.],
                                [ 40.,   0.],
                                [ 40.,  90.],
                                [ 40., 180.],
                                [ 40., 270.],
                                [ 40., 360.],
                                [ 50.,   0.],
                                [ 50.,  90.],
                                [ 50., 180.],
                                [ 50., 270.],
                                [ 50., 360.],
                                [ 60.,   0.],
                                [ 60.,  90.],
                                [ 60., 180.],
                                [ 60., 270.],
                                [ 60., 360.],
                                [ 70.,   0.],
                                [ 70.,  90.],
                                [ 70., 180.],
                                [ 70., 270.],
                                [ 70., 360.],
                                [ 80.,   0.],
                                [ 80.,  90.],
                                [ 80., 180.],
                                [ 80., 270.],
                                [ 80., 360.],
                                [ 90.,   0.],
                                [ 90.,  90.],
                                [ 90., 180.],
                                [ 90., 270.],
                                [ 90., 360.]])

def interpolate_to_lat_lon(fov_lat_lon, global_model, global_model_lat_lon):
    """
    fov_lat_lon : a array of [lattitude, longitude]

    Interpolate the input atmospheric model to the field of view averaging
    lattitudes and longitudes

    Snapped the global model for various physical quantities to the chosen
    set of locations on the planet with specified lattitudes and longitudes
    using bilinear interpolation.
    """
    # NLOC x NLAYER at desired lattitude and longitudesy
    NLOCFOV = fov_lat_lon.shape[0] # output

    # add an extra data point for the periodic longitude

    # global_model_lat_lon = np.append(global_model_lat_lon,)

    fov_model_output =  np.zeros(NLOCFOV)

    """make sure there is a point at lon = 0"""
    lat_grid = global_model_lat_lon[:,0]
    lon_grid = global_model_lat_lon[:,1]

    for iloc, location in enumerate(fov_lat_lon):
        lat = location[0]
        lon = location[1]

        if lat > np.max(lat_grid):
            lat = np.max(lat_grid)
        if lat < np.min(lat_grid):
            lat = np.min(lat_grid) + 1e-3
        if lon > np.max(lon_grid):
            lon = np.max(lon_grid)
        if lon < np.min(lon_grid):
            lon = np.min(lon_grid) + 1e-3

        lat_index_hi = np.where(lat_grid >= lat)[0][0]
        lat_index_low = np.where(lat_grid < lat)[0][-1]
        lon_index_hi = np.where(lon_grid >= lon)[0][0]
        lon_index_low = np.where(lon_grid < lon)[0][-1]

        lat_hi = lat_grid[lat_index_hi]
        lat_low = lat_grid[lat_index_low]
        lon_hi = lon_grid[lon_index_hi]
        lon_low = lon_grid[lon_index_low]

        Q11 = global_model[lat_index_low, lon_index_low]
        Q12 = global_model[lat_index_hi, lon_index_low]
        Q22 = global_model[lat_index_hi, lon_index_hi]
        Q21 = global_model[lat_index_low, lon_index_hi]

        fxy1 = (lon_hi-lon)/(lon_hi-lon_low)*Q11 + (lon-lon_low)/(lon_hi-lon_low)*Q21
        fxy2 = (lon_hi-lon)/(lon_hi-lon_low)*Q21 + (lon-lon_low)/(lon_hi-lon_low)*Q22
        fxy = (lat_hi-lat)/(lat_hi-lat_low)*fxy1 + (lat-lat_low)/(lat_hi-lat_low)*fxy2

        fov_model_output[iloc] = fxy


    return fov_model_output


def disc_avarage(wave_grid,wav,TP_grid_interped):
    """

    """





    disc_spec = None
    return disc_spec
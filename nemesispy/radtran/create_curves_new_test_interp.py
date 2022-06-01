import sys

# from nemesispy.radtran.disc_benchmarking import NLAYER
sys.path.append('/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/')
import matplotlib.pyplot as plt
import numpy as np
from nemesispy.data.constants import R_SUN, R_JUP_E, AMU, AU, M_JUP, R_JUP
from nemesispy.radtran.utils import calc_mmw
from nemesispy.radtran.models import Model2
from nemesispy.radtran.read import read_kls
from nemesispy.radtran.radiance import calc_radiance, calc_planck
from nemesispy.radtran.read import read_cia
from nemesispy.radtran.trig import gauss_lobatto_weights
from nemesispy.radtran.trig import interpvivien_point
from nemesispy.radtran.forward_model import ForwardModel
import time
# from nemesispy.radtran.runner import interpolate_to_lat_lon

### Opacity data
lowres_files = ['/Users/jingxuanyang/ktables/h2owasp43.kta',
'/Users/jingxuanyang/ktables/cowasp43.kta',
'/Users/jingxuanyang/ktables/co2wasp43.kta',
'/Users/jingxuanyang/ktables/ch4wasp43.kta']
cia_file_path \
    ='/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/cia/exocia_hitran12_200-3800K.tab'

### Reference Planet Input
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m
gas_id = np.array([  1, 2,  5,  6, 40, 39])
iso_id = np.array([0, 0, 0, 0, 0, 0])
NLAYER = 20

################################################################################
################################################################################
# Read GCM data
################################################################################
################################################################################
# Read GCM data
from nemesispy.radtran.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,hvmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,hvmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase)


lon = 361
lat = 87.188
old_lon = lon - 180
press = np.geomspace(pv[0],pv[-1],NLAYER)



interped_T, interped_VMR = interpvivien_point(XLON=lon, XLAT=lat, XP=np.geomspace(20e5,1,20),
    VP=pv,VT=tmap,VVMR=vmrmap,
    global_model_longitudes=xlon,
    global_model_lattitudes=xlat)

print(interped_T)
"""
print('interped_T ',interped_T)
# print('diag',np.min(tmap))
# print('interped_VMR ',interped_VMR)

old_T = interpolate_to_lat_lon(np.array([[old_lon,lat],[old_lon,lat]]), global_model=tmap,
            global_model_longitudes=xlon, global_model_lattitudes=xlat)
print('old_T ', old_T[0])

rdiff  = (interped_T-old_T[0])/old_T[0]
print('rdiff',np.round(max(rdiff),5),np.round(min(rdiff),5))
"""
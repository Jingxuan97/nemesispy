import sys

# from nemesispy.radtran.disc_benchmarking import NLAYER
sys.path.append('/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/')
import matplotlib.pyplot as plt
import numpy as np
from nemesispy.data.constants import R_SUN, R_JUP_E, AMU, AU, M_JUP, R_JUP
from nemesispy.radtran.utils import calc_mmw
from nemesispy.radtran.models import Model2
from nemesispy.radtran.path import calc_layer # average
from nemesispy.radtran.read import read_kls
from nemesispy.radtran.radiance import calc_radiance, calc_planck
from nemesispy.radtran.read import read_cia
from nemesispy.radtran.trig import gauss_lobatto_weights, interpolate_to_lat_lon
from nemesispy.radtran.trig2 import interpvivien_point
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
f = open('process_vivien.txt')
vivien_gcm = f.read()
f.close()
vivien_gcm = vivien_gcm.split()
vivien_gcm = [float(i) for i in vivien_gcm]

### Parse GCM data
iread = 152
nlon = 64
nlat = 32
npv = 53
xlon = np.array([-177.19  , -171.56  , -165.94  , -160.31  , -154.69  , -149.06 ,
       -143.44  , -137.81  , -132.19  , -126.56  , -120.94  , -115.31  ,
       -109.69  , -104.06  ,  -98.438 ,  -92.812 ,  -87.188 ,  -81.562 ,
        -75.938 ,  -70.312 ,  -64.688 ,  -59.062 ,  -53.438 ,  -47.812 ,
        -42.188 ,  -36.562 ,  -30.938 ,  -25.312 ,  -19.688 ,  -14.062 ,
         -8.4375,   -2.8125,    2.8125,    8.4375,   14.062 ,   19.688 ,
         25.312 ,   30.938 ,   36.562 ,   42.188 ,   47.812 ,   53.438 ,
         59.062 ,   64.688 ,   70.312 ,   75.938 ,   81.562 ,   87.188 ,
         92.812 ,   98.438 ,  104.06  ,  109.69  ,  115.31  ,  120.94  ,
        126.56  ,  132.19  ,  137.81  ,  143.44  ,  149.06  ,  154.69  ,
        160.31  ,  165.94  ,  171.56  ,  177.19  ])
xlat = np.array([-87.188 , -81.562 , -75.938 , -70.312 , -64.688 , -59.062 ,
       -53.438 , -47.812 , -42.188 , -36.562 , -30.938 , -25.312 ,
       -19.688 , -14.062 ,  -8.4375,  -2.8125,   2.8125,   8.4375,
        14.062 ,  19.688 ,  25.312 ,  30.938 ,  36.562 ,  42.188 ,
        47.812 ,  53.438 ,  59.062 ,  64.688 ,  70.312 ,  75.938 ,
        81.562 ,  87.188 ])
pv = np.array([1.7064e+02, 1.2054e+02, 8.5152e+01, 6.0152e+01, 4.2492e+01,
       3.0017e+01, 2.1204e+01, 1.4979e+01, 1.0581e+01, 7.4747e+00,
       5.2802e+00, 3.7300e+00, 2.6349e+00, 1.8613e+00, 1.3148e+00,
       9.2882e-01, 6.5613e-01, 4.6350e-01, 3.2742e-01, 2.3129e-01,
       1.6339e-01, 1.1542e-01, 8.1532e-02, 5.7595e-02, 4.0686e-02,
       2.8741e-02, 2.0303e-02, 1.4342e-02, 1.0131e-02, 7.1569e-03,
       5.0557e-03, 3.5714e-03, 2.5229e-03, 1.7822e-03, 1.2589e-03,
       8.8933e-04, 6.2823e-04, 4.4379e-04, 3.1350e-04, 2.2146e-04,
       1.5644e-04, 1.1051e-04, 7.8066e-05, 5.5146e-05, 3.8956e-05,
       2.7519e-05, 1.9440e-05, 1.3732e-05, 9.7006e-06, 6.8526e-06,
       4.8408e-06, 3.4196e-06, 2.4156e-06])*1e5

pvmap = np.zeros((nlon,nlat,npv))
for ilon in range(nlon):
    for ilat in range(nlat):
        pvmap[ilon,ilat,:] = pv

# tmp = np.zeros((7,npv))
tmap = np.zeros((nlon,nlat,npv))
co2map = np.zeros((nlon,nlat,npv))
h2map = np.zeros((nlon,nlat,npv))
hemap = np.zeros((nlon,nlat,npv))
ch4map = np.zeros((nlon,nlat,npv))
comap = np.zeros((nlon,nlat,npv))
h2omap = np.zeros((nlon,nlat,npv))

for ilon in range(nlon):
    for ilat in range(nlat):
        for ipv in range(npv):
            tmap[ilon,ilat,ipv] = vivien_gcm[iread]
            h2omap[ilon,ilat,ipv] = vivien_gcm[iread+6]
            co2map[ilon,ilat,ipv] = vivien_gcm[iread+1]
            comap[ilon,ilat,ipv] = vivien_gcm[iread+5]
            ch4map[ilon,ilat,ipv] = vivien_gcm[iread+4]
            hemap[ilon,ilat,ipv] = vivien_gcm[iread+3]
            h2map[ilon,ilat,ipv] = vivien_gcm[iread+2]
            iread+=7

vmrmap = np.zeros((nlon,nlat,npv,6))
for ilon in range(nlon):
    for ilat in range(nlat):
        for ipv in range(npv):
            vmrmap[ilon,ilat,ipv,0] = h2omap[ilon,ilat,ipv]
            vmrmap[ilon,ilat,ipv,1] = co2map[ilon,ilat,ipv]
            vmrmap[ilon,ilat,ipv,2] = comap[ilon,ilat,ipv]
            vmrmap[ilon,ilat,ipv,3] = ch4map[ilon,ilat,ipv]
            vmrmap[ilon,ilat,ipv,4] = hemap[ilon,ilat,ipv]
            vmrmap[ilon,ilat,ipv,5] = h2map[ilon,ilat,ipv]


lon = 200
lat = 0
old_lon = lon - 180
press = np.geomspace(pv[0],pv[-1],NLAYER)



interped_T, interped_VMR = interpvivien_point(XLON=lon, XLAT=lat, XP=pv,
    VP=pv,VT=tmap,VVMR=vmrmap,
    global_model_longitudes=xlon,
    global_model_lattitudes=xlat)


print('interped_T ',interped_T)
print('diag',np.min(tmap))
# print('interped_VMR ',interped_VMR)

old_T = interpolate_to_lat_lon(np.array([[old_lon,lat],[old_lon,lat]]), global_model=tmap,
            global_model_longitudes=xlon, global_model_lattitudes=xlat)
print('old_T ', old_T[0])

rdiff  = (interped_T-old_T[0])/old_T[0]
print('rdiff',rdiff)
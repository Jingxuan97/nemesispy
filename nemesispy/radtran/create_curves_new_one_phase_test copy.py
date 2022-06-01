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

lowres_files = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/testktables/h2owasp43.kta',
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/testktables/cowasp43.kta',
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/testktables/co2wasp43.kta',
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/testktables/ch4wasp43.kta']

### Reference Planet Input
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m
gas_id = np.array([  1, 2,  5,  6, 40, 39])
iso_id = np.array([0, 0, 0, 0, 0, 0])
NLAYER = 88

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
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)

"""
lon = 0
lat = 0
press = np.geomspace(pv[0],pv[-1],NLAYER)



interped_T, interped_VMR = interpvivien_point(XLON=lon, XLAT=lat, XP=pv,
    VP=pv,VT=tmap,VVMR=vmrmap,
    global_model_longitudes=xlon,
    global_model_lattitudes=xlat)


print('interped_T ',interped_T)
print('diag',np.min(tmap))
# print('interped_VMR ',interped_VMR)
"""

"""
old_T = interpolate_to_lat_lon(np.array([[0,lat],[0,lat]]), global_model=tmap,
            global_model_longitudes=xlon, global_model_lattitudes=xlat)
print('old_T ', old_T[0])

rdiff  = (interped_T-old_T[0])/old_T[0]
print('rdiff',rdiff)
"""


################################################################################
################################################################################
start = time.time()
### Set up forward model
FM = ForwardModel()
FM.set_planet_model(M_plt=M_plt,R_plt=R_plt,gas_id_list=gas_id,iso_id_list=iso_id,
    NLAYER=NLAYER)
FM.set_opacity_data(kta_file_paths=lowres_files, cia_file_path=cia_file_path)

### Code to actually simulate a phase curve
wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875,
       1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6   ,
       4.5   ])
phase_grid = np.array([ 22.5,  45. ,  67.5,  90. , 112.5, 135. , 157.5, 180. , 202.5,
       225. , 247.5, 270. , 292.5, 315. , 337.5])
nwave = len(wave_grid)
nphase = len(phase_grid)

my_gcm_phase_by_wave = np.zeros((nphase,nwave))
my_gcm_wave_by_phase = np.zeros((nwave,nphase))

wasp43_spec = np.array([3.341320e+25, 3.215455e+25, 3.101460e+25, 2.987110e+25,
       2.843440e+25, 2.738320e+25, 2.679875e+25, 2.598525e+25,
       2.505735e+25, 2.452230e+25, 2.391140e+25, 2.345905e+25,
       2.283720e+25, 2.203690e+25, 2.136015e+25, 1.234010e+24,
       4.422200e+23])

wasp43_spec = np.array([3.341320e+25, 3.215455e+25, 3.101460e+25, 2.987110e+25,
       2.843440e+25, 2.738320e+25, 2.679875e+25, 2.598525e+25,
       2.505735e+25, 2.452230e+25, 2.391140e+25, 2.345905e+25,
       2.283720e+25, 2.203690e+25, 2.136015e+25, 1.234010e+24,
       4.422200e+23])

"""
phasenumber = 7
nmu = 5
phase = phase_grid[phasenumber]
P_model = np.geomspace(20e5,1,NLAYER)
P_model = pv
one_phase =  FM.test_disc_spectrum(phase=phase, nmu=nmu, P_model = pv,
    global_model_P_grid=pv,
    global_T_model=tmap_mod, global_VMR_model=vmrmap_mod_new,
    global_model_longitudes=xlon,
    global_model_lattitudes=xlat,
    solspec=wasp43_spec)

fig, axs = plt.subplots(nrows=2,ncols=1,sharex=True,
    dpi=800)
axs[0].set_title('phase = {}'.format(phase))
axs[0].plot(wave_grid,one_phase,color='b',label='Python')
axs[0].scatter(wave_grid,kevin_phase_by_wave[phasenumber,:,0],color='r',marker='+',label='Data')
axs[0].plot(wave_grid,pat_phase_by_wave[phasenumber,:],color ='k',label='Fortran')
axs[0].legend()
axs[0].grid()

diff = (one_phase - pat_phase_by_wave[phasenumber,:])/one_phase
axs[1].scatter(wave_grid,diff,marker='.',color='b')
axs[1].grid()
print(diff)
"""
for iphase in range(nphase):
    phasenumber = iphase
    nmu = 5
    phase = phase_grid[phasenumber]
    P_model = np.geomspace(20e5,1,NLAYER)
    P_model = pv
    one_phase =  FM.test_disc_spectrum(phase=phase, nmu=nmu, P_model = pv,
        global_model_P_grid=pv,
        global_T_model=tmap, global_VMR_model=vmrmap_mod_new,
        global_model_longitudes=xlon,
        global_model_lattitudes=xlat,
        solspec=wasp43_spec)

    fig, axs = plt.subplots(nrows=2,ncols=1,sharex=True,
        dpi=800)
    axs[0].set_title('phase = {}'.format(phase))
    axs[0].plot(wave_grid,one_phase,color='b',label='Python')
    axs[0].scatter(wave_grid,kevin_phase_by_wave[phasenumber,:,0],color='r',marker='+',label='Data')
    axs[0].plot(wave_grid,pat_phase_by_wave[phasenumber,:],color ='k',label='Fortran')
    axs[0].legend(loc='upper left')
    axs[0].grid()
    axs[0].set_ylabel('Flux ratio')

    diff = (one_phase - pat_phase_by_wave[phasenumber,:])/one_phase
    axs[1].scatter(wave_grid,diff,marker='.',color='b')
    axs[1].grid()
    axs[1].set_ylabel('Relative diff')
    axs[1].set_xlabel('Wavelength (Micron)')
    print(iphase,diff)
    plt.tight_layout()


    plt.savefig('good_discav_planet{}.pdf'.format(iphase),dpi=800)


"""
plt.plot(wave_grid,one_phase)
plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.savefig('discav_planet.pdf',dpi=400)
plt.show()
"""
"""
diff_phase = np.array([-0.00897619,-0.0079478,-0.01448685,-0.02673921,-0.01743425,-0.0104249
,-0.00883205,-0.00889048,-0.0146716,-0.04906669,-0.03530353,-0.01973727
,-0.01512038,-0.06898747,-0.10909267,0.01644151,-0.47769273])
diff_phase = np.array([-0.00692144,-0.00592898,-0.01153512,-0.02278401,-0.01551818,-0.00892093
,-0.00617974,-0.0057484,-0.00978439,-0.03509447,-0.02710494,-0.01532344
,-0.01113158,-0.05359842,-0.0870141,0.01274438,-0.44370787])
diff_phase = np.array([-0.00487615,-0.00397899,-0.00826619,-0.01778308,-0.01281296,-0.00702675
,-0.00409304,-0.00380911,-0.00653695,-0.02390706,-0.01983174,-0.01127137
,-0.00745722,-0.03756856,-0.06265161,0.0089494,-0.40989064])
diff_phase = np.array([-0.00328106,-0.00249635,-0.00552885,-0.01310958,-0.01008617,-0.0052506
,-0.00274792,-0.00278629,-0.00473583,-0.01679774,-0.01472545,-0.00825721
,-0.00475286,-0.02482127,-0.04220652,0.00589352,-0.38194794])
diff_phase = np.array([-0.00217411,-0.00147916,-0.00357032,-0.00959784,-0.00795998,-0.00390432
,-0.00192715,-0.00221737,-0.0037494,-0.01251876,-0.01128043,-0.00608403
,-0.00287799,-0.01562332,-0.02684033,0.00371842,-0.36121802])
diff_phase = np.array([-0.00153432,-0.00087828,-0.00246685,-0.00777359,-0.00684187,-0.00313531
,-0.00144625,-0.00187698,-0.00317968,-0.00991839,-0.00898307,-0.00459327
,-0.0016945,-0.00982601,-0.01725275,0.00232647,-0.34249361])
diff_phase = np.array([-0.00141254,-0.00072943,-0.00238296,-0.00809216,-0.00702432,-0.00310603
,-0.00119603,-0.00158848,-0.00275202,-0.00833923,-0.0075282,-0.00370514
,-0.00112047,-0.00779689,-0.01485397,0.00163519,-0.32265839])
diff_phase = np.array([-0.00194802,-0.00118921,-0.00358906,-0.01077183,-0.00867975,-0.00399572
,-0.0011408,-0.00122568,-0.00232199,-0.00776189,-0.0070211,-0.00350786
,-0.00134058,-0.01065189,-0.0210127,0.0021601,-0.29992858])
diff_phase = np.array([-0.00325559,-0.0023713,-0.00607724,-0.01535345,-0.01142028,-0.00570124
,-0.0014438,-0.00093712,-0.00208291,-0.00859962,-0.00776041,-0.00424965
,-0.00262647,-0.01882696,-0.03577669,0.00399728,-0.27925146])
diff_phase = np.array([-0.00553691,-0.00451769,-0.00986352,-0.02126319,-0.01480625,-0.00808765
,-0.00250588,-0.00109645,-0.00248152,-0.01186908,-0.01048487,-0.00645431
,-0.00540269,-0.03294019,-0.05921289,0.00685676,-0.26286919])
diff_phase = np.array([-0.0085749,-0.00744401,-0.01432644,-0.0272814,-0.01804407,-0.0106386
,-0.00468047,-0.00217572,-0.00420412,-0.01944463,-0.01631033,-0.01062873
,-0.00983847,-0.05244998,-0.08906534,0.01067766,-0.26542991])
diff_phase = np.array([-0.01138336,-0.01026986,-0.01806486,-0.03160991,-0.02004642,-0.01245333
,-0.00811397,-0.00513443,-0.00862209,-0.03427784,-0.02642016,-0.01680565
,-0.01539663,-0.07386831,-0.11940607,0.01503802,-0.28660373])
diff_phase = np.array([-0.01262382,-0.01158029,-0.0195868,-0.03309622,-0.02045453,-0.01297814
,-0.01154775,-0.01022279,-0.01620672,-0.05505196,-0.03889959,-0.02310254
,-0.02005143,-0.0891424,-0.13873669,0.01904425,-0.33494666])
diff_phase = np.array([-0.01224192,-0.01117031,-0.01911681,-0.03255863,-0.01996805,-0.01257524
,-0.01272445,-0.01355722,-0.02146815,-0.06684042,-0.04485995,-0.02548874
,-0.02120538,-0.09269588,-0.14282221,0.02103979,-0.41586263])
diff_phase = np.array([-0.0114348,-0.01034621,-0.01800492,-0.03119657,-0.01933932,-0.01202766
,-0.01203511,-0.01312029,-0.02121574,-0.06661734,-0.0445168,-0.02467299
,-0.02003158,-0.08874785,-0.136799,0.02148724,-0.4658371,])
"""
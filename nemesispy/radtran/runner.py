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
from nemesispy.radtran.trig import gauss_lobatto_weights, interpolate_to_lat_lon

class runner:

    def __init__(self, gas_file, cia_file):
        self.gas_file = gas_file
        self.cia_file = cia_file


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
    fov_lat_lon(NLOC,2) : a array of [lattitude, longitude]

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


"""LAT        LON         StarZen      EmiZen      StarAzi  Weight
    0.0000     18.4349    161.5651     63.4349      0.0000  0.07701087
   35.1970     11.8202    143.1155     63.4349     49.3939  0.15402173
   61.8232    333.7203    115.0488     63.4349     50.2848  0.15402173
   50.5081    269.6843     89.7992     63.4349     37.5965  0.19146977
    0.0000    251.5651     71.5651     63.4349      0.0000  0.11445891
    0.0000    315.0000    135.0000      0.0000    180.0000  0.30901699
"""

from nemesispy.radtran.radiance import calc_radiance
def weighted_avaraged_spectrum(wave_grid,
    k_gas_w_g_p_t, P_grid, T_grid, del_g,
    emi_zen_list, weight_list,
    U_layer_list,P_layer_list,VMR_layer_list,
    ):
    """
    emi_zen_list(NLOC) :
        A array of emission zenith angles.
    """
    NLOC = len(emi_zen_list)

    for iloc, iangle in range(NLOC):
        U_layer = U_layer_list[iloc]
        P_layer = P_layer_list[iloc]
        T_layer = T_layer_list[iloc]
        VMR_layer = VMR_layer_list[iloc]

        ispec = calc_radiance(wave_grid, U_layer, P_layer, T_layer,
            VMR_layer, k_gas_w_g_p_t,
            P_grid, T_grid, del_g, ScalingFactor, RADIUS, solspec,
            k_cia, ID, NU_GRID, CIA_TEMPS, DEL_S)



from nemesispy.radtran.path import calc_layer
def fm(wave_grid, StarSpectrum, k_gas_w_g_p_t, P_grid, T_grid, del_g,
    k_cia, CIA_NU_GRID, CIA_TEMPS,
    planet_radius, H_model, P_model, T_model, VMR_model, ID, NLAYER, path_angle,
    H_0=0.0, NSIMPS=101,layer_type=1,custom_path_angle=0.0, custom_H_base=None,
    custom_P_base=None):

    H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,ScalingFactor,del_S\
    = calc_layer(planet_radius, H_model, P_model, T_model, VMR_model, ID, NLAYER,
        path_angle, H_0=H_0, NSIMPS=NSIMPS, layer_type=layer_type,
        custom_path_angle=custom_path_angle, custom_H_base=custom_H_base,
        custom_P_base=custom_P_base)

    SPECOUT = calc_radiance(wave_grid, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
        P_grid, T_grid, del_g, ScalingFactor=ScalingFactor,
        RADIUS=planet_radius, solspec=StarSpectrum,
        k_cia=k_cia,ID=ID,NU_GRID=CIA_NU_GRID,CIA_TEMPS=CIA_TEMPS, DEL_S=del_S)

    return None


from nemesispy.radtran.read import read_kls, read_cia
class RetrievalModel():

    def read_data()

import Model2

from nemesispy.radtran.read import read_kls, read_cia
class FowardModel_provisional():

    def __init__(self):

        # set by set_planet_model
        self.is_planet_loaded = False
        self.T_star = 0
        self.R_star = 0
        self.M_plt = 0
        self.R_plt = 0
        self.SMA = 0
        self.star_spectrum = []

        self.gas_id_list = []
        self.gas_iso_list = []
        self.NMODEL = 0

        self.P_range = []

        # set by read_opacity_data
        self.is_data_loaded = False
        self.g_ord = []
        self.del_g = []
        self.P_grid = []
        self.T_grid = []
        self.k_gas_w_g_p_t = []

        self.cia_nu_grid = []
        self.cia_T_grid = []
        self.k_cia_pair_t_w = []

    def read_opacity_data(self, kta_file_paths, cia_file_path):
        gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
            k_gas_w_g_p_t = read_kls(kta_file_paths)
        """
        Read gas ktables and cia opacity files and store as class attributes.
        """
        cia_nu_grid, cia_T_grid, k_cia_pair_t_w = read_cia(cia_file_path)

        self.wave_grid = wave_grid
        self.g_ord = g_ord
        self.del_g = del_g
        self.P_grid = P_grid
        self.T_grid = T_grid
        self.k_gas_w_g_p_t = k_gas_w_g_p_t

        self.cia_nu_grid = cia_nu_grid
        self.cia_T_grid = cia_T_grid
        self.k_cia_pair_t_w = k_cia_pair_t_w

        self.is_data_loaded = True

    def set_planet_model(self,T_star,R_star,M_plt,R_plt,SMA,NMODEL,P_max,P_min):
        """
        Set the basic system parameters for running the
        """
        self.T_star = T_star
        self.R_star = R_star
        self.M_plt = M_plt
        self.R_plt = R_plt
        self.SMA = SMA
        self.NMODEL = NMODEL
        self.P_range = np.logspace(np.log10(P_max),np.log10(P_min),NMODEL)
        self.is_planet_loaded = True

    def set_atmosphere_model(self, kappa, gamma1, gamma2, alpha, T_irr,
        VMR_H2O,VMR_CO2,VMR_CO,VMR_CH4,H2ratio=0.85):

        np.logspace(np.log10(20/1.01325),np.log10(1e-3/1.01325),self.NMODEL)*1e5

        VMR_H2O = np.ones(self.NMODEL) * VMR_H2O
        VMR_CO2 = np.ones(self.NMODEL) * VMR_CO2
        VMR_CO = np.ones(self.NMODEL) * VMR_CO
        VMR_CH4 = np.ones(self.NMODEL) * VMR_CH4
        VMR_He = (np.ones(self.NMODEL)-VMR_H2O-VMR_CO2-VMR_CO-VMR_CH4)*(1-H2ratio)
        VMR_H2 = (np.ones(self.NMODEL)-VMR_H2O-VMR_CO2-VMR_CO-VMR_CH4)*H2ratio

        NVMR = len(self.ID)
        VMR_model = np.zeros((self.NMODEL,NVMR))
        VMR_model[:,0] = VMR_H2O
        VMR_model[:,1] = VMR_CO2
        VMR_model[:,2] = VMR_CO
        VMR_model[:,3] = VMR_CH4
        VMR_model[:,4] = VMR_He
        VMR_model[:,5] = VMR_H2

        mmw = calc_mmw(self.ID, VMR_model[0,:])
        atm = Model2(self.T_star, self.R_star, self.M_plt, self.R_plt, self.P_range,
        mmw, kappa, gamma1, gamma2, alpha, T_irr)

        H_model = atm.height()
        P_model = atm.pressure()
        T_model = atm.temperature()
        return H_model, P_model, T_model, VMR_model

    def cal_layer_properties(self,H_model, P_model, T_model, VMR_model, ID, Nlayer,
        path_angle, H_0=0.0, NSIMPS=101, layer_type=1,
        custom_path_angle=0.0, custom_H_base=None, custom_P_base=None):

        H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,ScalingFactor,del_S\
            = calc_layer(self.R_plt, H_model, P_model, T_model, VMR_model, ID,
            Nlayer, path_angle, H_0=H_0, NSIMPS=NSIMPS, layer_type=layer_type,
            custom_path_angle=custom_path_angle, custom_H_base=custom_H_base,
            custom_P_base=custom_P_base)

        return H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,ScalingFactor,del_S

    def calc_point_radiance(self, wave_grid, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
        P_grid, T_grid, del_g, ScalingFactor,
        RADIUS, solspec, k_cia,ID, cia_nu_grid, cia_T_grid, DEL_S):
        specout = calc_radiance(wave_grid, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
            P_grid, T_grid, del_g, ScalingFactor,RADIUS, solspec,
            k_cia,ID,cia_nu_grid,cia_T_grid, DEL_S)
        return specout

    def run_forward_model(self,kappa, gamma1, gamma2, alpha, T_irr,
        VMR_H2O,VMR_CO2,VMR_CO,VMR_CH4,Nlayer,path_angle,H_0=0.0, NSIMPS=101,
        layer_type=1, custom_path_angle=0.0, custom_H_base=None, custom_P_base=None,
        H2ratio=0.85):

        H_model, P_model, T_model, VMR_model\
            = self.set_atmosphere_model(kappa, gamma1, gamma2, alpha, T_irr,
        VMR_H2O,VMR_CO2,VMR_CO,VMR_CH4,H2ratio=H2ratio)

        H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,ScalingFactor,del_S\
            = self.cal_layer_properties(H_model, P_model, T_model, VMR_model,
            ID=self.gas_id_list, Nlayer=Nlayer, path_angle=path_angle,H_0=H_0,
            NSIMPS=NSIMPS, layer_type=layer_type,custom_path_angle=custom_path_angle,
            custom_H_base=custom_H_base,custom_P_base=custom_P_base)

        specout = calc_radiance(self.wave_grid, U_layer, P_layer, T_layer, VMR_layer,
            self.k_cia_pair_t_w, self.P_grid, self.T_grid, self.del_g, ScalingFactor,
            self.R_plt, self.star_spectrum, self.k_cia_pair_t_w, self.gas_id_list,
            self.cia_nu_grid, self.cia_T_grid, del_S)

        return specout

    def



class ForwardModel():

    def __init__(self):

        # planet and planetary system data
        self.M_plt = None
        self.R_plt = None
        self.M_star = None # currently not used
        self.R_star = None
        self.T_star = None
        self.semi_major_axis = None
        self.NLAYER = None
        self.is_planet_model_set = False

        # opacity data
        self.gas_id_list = None
        self.iso_id_list = None
        self.wave_grid = None
        self.g_ord = None
        self.del_g = None
        self.k_table_P_grid = None
        self.k_table_T_grid = None
        self.k_gas_w_g_p_t = None
        self.cia_nu_grid = None
        self.cia_T_grid = None
        self.k_cia_pair_t_w = None
        self.is_opacity_data_set = False


    def set_planet_model(self, M_plt, R_plt, R_star, T_star, semi_major_axis,
        gas_id_list, iso_id_list, NLAYER):
        """
        Set the basic system parameters for running the
        """
        self.M_plt = M_plt
        self.R_plt = R_plt
        self.R_star = R_star
        self.T_star = T_star
        self.semi_major_axis = semi_major_axis
        self.gas_id_list = gas_id_list
        self.iso_id_list = iso_id_list
        self.NLAYER = NLAYER

        self.is_planet_model_set = True

    def set_opacity_data(self, kta_file_paths, cia_file_path):
        """
        Read gas ktables and cia opacity files and store as class attributes.
        """
        gas_id_list, iso_id_list, wave_grid, g_ord, del_g, k_table_P_grid,\
            k_table_T_grid, k_gas_w_g_p_t = read_kls(kta_file_paths)


        #self.gas_id_list = gas_id_list
        #self.iso_id_list = iso_id_list
        self.wave_grid = wave_grid
        self.g_ord = g_ord
        self.del_g = del_g
        self.k_table_P_grid = k_table_P_grid
        self.k_table_T_grid = k_table_T_grid
        self.k_gas_w_g_p_t = k_gas_w_g_p_t

        cia_nu_grid, cia_T_grid, k_cia_pair_t_w = read_cia(cia_file_path)
        self.cia_nu_grid = cia_nu_grid
        self.cia_T_grid = cia_T_grid
        self.k_cia_pair_t_w = k_cia_pair_t_w

        self.is_opacity_data_set = True

    def calc_point_spectrum(self, H_model, P_model, T_model, VMR_model, path_angle,
        solspec=None):
        """
        First get layer properties from model inputs
        Then calculate the spectrum at a single point on the disc.
        """

        H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,scale,del_S\
            = calc_layer(self.R_plt, H_model, P_model, T_model, VMR_model,
            self.gas_id_list, self.NLAYER, path_angle, layer_type=1, H_0=0.0, NSIMPS=101)
        if solspec == None:
            solspec = np.ones(len(self.wave_grid))
        point_spectrum = calc_radiance(self.wave_grid, U_layer, P_layer, T_layer,
            VMR_layer, self.k_gas_w_g_p_t, self.k_table_P_grid,
            self.k_table_T_grid, self.del_g, ScalingFactor=scale,
            RADIUS=self.R_plt, solspec=solspec, k_cia=self.k_cia_pair_t_w,
            ID=self.gas_id_list, cia_nu_grid=self.cia_nu_grid,
            cia_T_grid=self.cia_T_grid,DEL_S=del_S)

        return point_spectrum

    def calc_disc_spectrum(self,phase,nmu,global_H_model,global_P_model,
        global_T_model,global_VMR_model,global_model_longitudes,
        global_model_lattitudes,solspec=None):

        nav, wav = gauss_lobatto_weights(phase, nmu)

        fov_longitudes = wav[1,:]
        fov_lattitudes = wav[0,:]
        fov_emission_angles = wav[3,:]
        fov_weights = wav[-1,:]

        # n_fov_loc = len(fov_longitudes)
        fov_locations = np.zeros((nav,2))
        fov_locations[:,0] = fov_longitudes
        fov_locations[:,1] = fov_lattitudes

        fov_H_model = interpolate_to_lat_lon(fov_locations, global_H_model,
            global_model_longitudes, global_model_lattitudes)

        fov_T_model = interpolate_to_lat_lon(fov_locations, global_T_model,
            global_model_longitudes, global_model_lattitudes)

        fov_VMR_model = interpolate_to_lat_lon(fov_locations, global_VMR_model,
            global_model_longitudes, global_model_lattitudes)

        disc_spectrum = np.zeros(len(self.wave_grid))

        for ilocation in range(nav):
            H_model = fov_H_model[ilocation]
            P_model = global_P_model[0]
            T_model = fov_T_model[ilocation]
            VMR_model = fov_VMR_model[ilocation]
            path_angle = fov_emission_angles[ilocation]
            weight = fov_weights[ilocation]
            point_spectrum = self.calc_point_spectrum(
                H_model, P_model, T_model, VMR_model, path_angle,
                solspec=solspec)
            disc_spectrum += point_spectrum * weight
        return disc_spectrum

    """TBD"""
    def run_point_spectrum(self, H_model, P_model, T_model,\
            VMR_model, path_angle, solspec=None):
        if self.is_planet_model_set == False:
            raise('Planet model has not been set yet')
        elif self.is_opacity_data_set == False:
            raise('Opacity data has not been set yet')
        point_spectrum = self.calc_point_spectrum(H_model, P_model, T_model,\
            VMR_model, path_angle, solspec=solspec)
        return point_spectrum

    def run_disc_spectrum(self):

        pass




class RetrievalModel():
    def __init__(self) -> None:
        pass

    def set_atmosphere_model(self,atm_model_parameters):
        pass






#     disc_spec = None
#     return disc_spec

# def emission_forward_model():
#     from nemesispy.radtran.read import read_kls
#     from nemesispy.radtran.read import read_cia

#     gas_id = []
#     gas_iso = []
#     gas_opacity_files = ''
#     cia_opacity_file = ''

#     gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
#         k_gas_w_g_p_t = read_kls(gas_opacity_files)
#     CIA_NU_GRID,CIA_TEMPS,K_CIA = read_cia(cia_file_path)


#     spectrum = []
#     return spectrum
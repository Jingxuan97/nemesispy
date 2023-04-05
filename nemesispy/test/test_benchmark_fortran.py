import unittest
import numpy as np
from nemesispy.radtran.forward_model import ForwardModel
from test_data.gcm_wasp_43b import gcm_lon_grid, \
    gcm_lat_grid, gcm_press_grid, tmap, vmrmap_mean, fortran_spectra
from nemesispy.data.helper import lowres_file_paths, cia_file_path

class TestBenchmarkAgainstFortran(unittest.TestCase):
    M_plt = 3.8951064000000004e+27 # kg
    R_plt = 74065.70 * 1e3 # m
    gas_id = np.array([  1, 2,  5,  6, 40, 39])
    iso_id = np.array([0, 0, 0, 0, 0, 0])
    NLAYER = 20
    P_model = np.geomspace(20e5,100,NLAYER)
    wave_grid = np.array(
        [1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875,
        1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6   ,
        4.5   ])
    phase_grid = np.array(
        [ 22.5,  45. ,  67.5,  90. , 112.5, 135. , 157.5, 180. , 202.5,
        225. , 247.5, 270. , 292.5, 315. , 337.5])
    stellar_spec = np.array(
        [3.341320e+25, 3.215455e+25, 3.101460e+25, 2.987110e+25,
        2.843440e+25, 2.738320e+25, 2.679875e+25, 2.598525e+25,
        2.505735e+25, 2.452230e+25, 2.391140e+25, 2.345905e+25,
        2.283720e+25, 2.203690e+25, 2.136015e+25, 1.234010e+24,
        4.422200e+23])
    nmu = 5
    FM = ForwardModel()
    FM.set_planet_model(
        M_plt=M_plt, R_plt=R_plt,
        gas_id_list=gas_id, iso_id_list=iso_id,
        NLAYER=NLAYER)
    FM.set_opacity_data(
        kta_file_paths=lowres_file_paths,
        cia_file_path=cia_file_path)
    def  test_gcm_spectra(self):
        phase_grid = self.phase_grid
        FM = self.FM
        for iphase, phase in enumerate(phase_grid):
            spectrum =  FM.calc_disc_spectrum(
                phase=phase, nmu=self.nmu, P_model = self.P_model,
                global_model_P_grid=gcm_press_grid,
                global_T_model=tmap,
                global_VMR_model=vmrmap_mean,
                mod_lon=gcm_lon_grid,
                mod_lat=gcm_lat_grid,
                solspec=self.stellar_spec)
            benchmark = fortran_spectra[iphase,:]
            residuals = abs((spectrum-benchmark)/benchmark)
            mean_diff = np.average(residuals)
            self.assertLess(mean_diff,0.01)


if __name__ == "__main__":
    unittest.main()
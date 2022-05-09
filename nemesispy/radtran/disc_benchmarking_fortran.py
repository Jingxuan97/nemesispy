import sys
sys.path.append('/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/')
import numpy as np
import matplotlib.pyplot as plt
import os
from nemesispy.radtran.disc_benchmarking_utils import phase_curve_fil,wasp43b_spx_dayside_single_angle_45
### Reference Opacity Data
spx_2zenith = """0.00000      0.00000      0.00000      1
17
7
    0.0000     40.9349    139.0651     63.4349      0.0000  0.06909830
   31.7175     35.7825    133.6367     63.4349     18.1075  0.13819660
   58.2825      9.2175    121.2613     63.4349     25.1995  0.13819660
   58.2825    305.7825    107.9026     63.4349     22.4869  0.13819660
   31.7175    279.2175     97.8314     63.4349     13.1237  0.13819660
    0.0000    274.0651     94.0651     63.4349      0.0000  0.06909830
    0.0000    337.5000    157.5000      0.0000    180.0000  0.30901699
1.14250  6.00000e-05  6.60000e-05
1.17750  5.50000e-05  6.10000e-05
1.21250  6.60000e-05  5.80000e-05
1.24750  8.60000e-05  5.60000e-05
1.28250  5.30000e-05  5.70000e-05
1.31750  9.00000e-05  5.30000e-05
1.35250  2.00000e-06  5.50000e-05
1.38750  2.90000e-05  5.20000e-05
1.42250  3.50000e-05  5.60000e-05
1.45750 -3.00000e-06  5.60000e-05
1.49250  2.40000e-05  5.60000e-05
1.52750 -3.00000e-06  5.50000e-05
1.56250  2.20000e-05  5.80000e-05
1.59750  4.80000e-05  5.80000e-05
1.63250  8.70000e-05  6.30000e-05
3.60000 -1.30000e-05  0.000103000
4.50000  9.50000e-05  0.000133000
"""
spx_5zenith = """      0.00000      0.00000      0.00000      1
      17
      47
      0.00000      57.9866      122.013      80.4866      0.00000   0.00377792
      12.5013      57.7533      121.394      80.4866      5.64680   0.00755584
      24.9841      56.9938      119.588      80.4866      10.8630   0.00755584
      37.4216      55.4883      116.741      80.4866      15.3093   0.00755584
      49.7604      52.6758      113.059      80.4866      18.7788   0.00755584
      61.8601      46.9855      108.768      80.4866      21.1844   0.00755584
      73.1854      32.6554      104.096      80.4866      22.5173   0.00755584
      80.4055      344.920      99.2612      80.4866      22.8081   0.00755584
      75.1711      287.725      94.4689      80.4866      22.1010   0.00755585
      64.1890      269.809      89.9168      80.4866      20.4451   0.00770491
      51.6919      262.964      85.6450      80.4866      17.7796   0.00785396
      38.8888      259.760      82.0460      80.4866      14.2391   0.00785396
      25.9677      258.093      79.3105      80.4866      9.95650   0.00785396
      12.9943      257.266      77.5972      80.4866      5.12522   0.00785397
      0.00000      257.013      77.0134      80.4866  1.36604e-05   0.00392698
      0.00000      38.9500      141.050      61.4500      0.00000    0.0106375
      12.1350      38.2351      140.168      61.4500      8.22016    0.0212750
      24.0926      35.9305      137.663      61.4500      15.3110    0.0212750
      35.6257      31.4870      133.880      61.4500      20.6139    0.0212750
      46.2952      23.7352      129.235      61.4500      23.9919    0.0212750
      55.2170      10.5941      124.107      61.4500      25.6041    0.0212750
      60.6914      349.992      118.820      61.4500      25.6963    0.0212750
      60.6914      325.008      113.642      61.4500      24.5005    0.0212750
      55.2170      304.406      108.805      61.4500      22.2093    0.0212750
      46.2952      291.265      104.512      61.4500      18.9851    0.0212750
      35.6257      283.513      100.949      61.4500      14.9794    0.0212750
      24.0926      279.069      98.2736      61.4500      10.3530    0.0212750
      12.1350      276.765      96.6130      61.4500      5.28993    0.0212750
      0.00000      276.050      96.0500      61.4500  1.36604e-05    0.0106375
      0.00000      19.8729      160.127      42.3729  1.36604e-05    0.0164610
      12.0206      18.4453      158.098      42.3729      18.4833    0.0329220
      23.3371      13.9286      153.023      42.3729      29.7266    0.0329220
      33.0410      5.70078      146.526      42.3729      34.1465    0.0329220
      39.8640      353.243      139.662      42.3729      34.2127    0.0329220
      42.3729      337.500      133.042      42.3729      31.5750    0.0329220
      39.8640      321.757      127.074      42.3729      27.1397    0.0329220
      33.0410      309.299      122.069      42.3729      21.4289    0.0329220
      23.3371      301.071      118.287      42.3729      14.7996    0.0329220
      12.0206      296.555      115.929      42.3729      7.55581    0.0329220
      0.00000      295.127      115.127      42.3729  1.36604e-05    0.0164610
      0.00000     0.641968      179.358      23.1420  0.000450792    0.0242898
      13.3563      356.574      166.219      23.1420      70.7822    0.0485796
      21.9486      345.024      153.639      23.1420      55.0517    0.0485796
      21.9486      329.976      143.424      23.1420      37.6451    0.0485796
      13.3563      318.426      136.708      23.1420      19.1491    0.0485796
      0.00000      314.358      134.358      23.1420  1.36604e-05    0.0242898
      0.00000      337.500      157.500      0.00000      180.000    0.0440346
      1.14250  7.40268e-05  6.60000e-05
      1.17750  0.000143689  6.10000e-05
      1.21250  0.000235543  5.80000e-05
      1.24750  0.000370310  5.60000e-05
      1.28250  0.000321902  5.70000e-05
      1.31750  0.000171065  5.30000e-05
      1.35250  8.60438e-05  5.50000e-05
      1.38750  1.85589e-05  5.20000e-05
      1.42250  0.000111874  5.60000e-05
      1.45750  4.77475e-05  5.60000e-05
      1.49250  0.000160166  5.60000e-05
      1.52750  9.82523e-05  5.50000e-05
      1.56250  0.000140665  5.80000e-05
      1.59750  0.000286794  5.80000e-05
      1.63250  0.000429466  6.30000e-05
      3.60000   0.00181714  0.000103000
      4.50000   0.00203622  0.000133000"""

lowres_files = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/h2o',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co2',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/ch4']
cia_file_path='/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/cia/exocia_hitran12_200-3800K.tab'

k_table_location = """/Users/jingxuanyang/ktables/h2owasp43.kta
/Users/jingxuanyang/ktables/cowasp43.kta
/Users/jingxuanyang/ktables/co2wasp43.kta
/Users/jingxuanyang/ktables/ch4wasp43.kta
"""
cia_file_name='exocia_hitran12_200-3800K.tab'

### Reference Constants
pi = np.pi
const = {
    'R_SUN': 6.95700e8,      # m solar radius
    'R_JUP': 7.1492e7,       # m nominal equatorial Jupiter radius (1 bar pressure level)
    'AU': 1.49598e11,        # m astronomical unit
    'k_B': 1.38065e-23,      # J K-1 Boltzmann constant
    'R': 8.31446,            # J mol-1 K-1 universal gas constant
    'G': 6.67430e-11,        # m3 kg-1 s-2 universal gravitational constant
    'N_A': 6.02214e23,       # Avagadro's number
    'AMU': 1.66054e-27,      # kg atomic mass unit
    'ATM': 101325,           # Pa atmospheric pressure
}

class Nemesis_api:

    def __init__(self, name, NLAYER, gas_id_list, iso_id_list, wave_grid):

        self.name = name
        self.NLAYER = NLAYER
        self.NMODEL = 0
        self.wave_grid = wave_grid
        self.NWAVE = len(wave_grid)

        self.input_spectrum = wasp43b_spx_dayside_single_angle_45
        self.stellar_spectrum =  'wasp43_stellar_newgrav.txt'
        # self.stellar_spectrum =  'wasp43_stellar_newgrav_fake.txt'

        """
        # planet and planetary system data
        self.M_plt = M_plt # currently not used / passed in gravity.dat
        self.R_plt = R_plt # currently not used / passed in gravity.dat
        self.M_star = 0 # currently not used
        self.R_star = 0 # currently not used
        self.T_star = 0 # currently not used
        self.semi_major_axis = 0 # currently not used
        self.is_planet_model_set = False
        """

        # opacity data
        self.k_table_location = k_table_location
        self.cia_file_name = cia_file_name
        self.gas_id_list = gas_id_list
        self.iso_id_list = iso_id_list
        self.NVMR = len(gas_id_list)

        """
        self.wave_grid = []
        self.g_ord = []
        self.del_g = []
        self.k_table_P_grid = []
        self.k_table_T_grid = []
        self.k_gas_w_g_p_t = []
        self.cia_nu_grid = []
        self.cia_T_grid = []
        self.k_cia_pair_t_w = []
        self.is_opacity_data_set = False
        """

        # nemesis configuration
        # set up .fla file
        self.INORMAL = 1   # wether ortho/para-H2 ratio is in eqlm, (0=eqm, 1=normal) #
        self.IRAY = 1      # Rayleigh optical depth calculation,1=gas giant,2=CO2 dominated,>2=N2,O2 dominated #
        self.IH2O = 0      # additional H2O continuum
        self.ICH4 = 0      # additional CH4 continuum
        self.IO3 = 0       # additional O3 continuum
        self.INH3 = 0      # additional NH3 continuum
        self.IPTF = 0      # CH4 partition function normal (0) or high-temperature (1) for Hot Jupiters.
        self.IMIE = 0      # phase function from hgphase*.dat, 1=from PHASEN.DAT
        self.IUV = 0       # additional UV opacity
        # set up .inp file
        self.ISPACE = 1          # 1=wavelength space (micron) #
        self.ISCAT = 0           # 0=thermal emission calculation #
        self.ILBL = 0            # 0=correlated-k
        self.WOFF = 0.0          # calibration error
        self.ENAME = 'noise.dat' # name of the file that contains the forward modelling error #
        self.NITER = -1          # NUMBER OF ITERATIONS of the retrieval model required, -1 is forward model calculation #
        self.PHILIMIT = 0.001    # percentage convergence limit; not used in nested sampling
        self.NSPEC = 1  # NSPEC=no of spectra to retrieve;
        self.IOFF = 1            # IOFF=index of first spectrum to fit
        self.LIN = 0             # 0=don't use previous retrievals to set atm prf
        self.IFORM = 1           # spectral unit: 0=radiance, 1=F_p/F_* (eclipse), 2=A_p/A_* (transit) #
        # set up .fil file
        self.filter_function = phase_curve_fil
        # set up .set file
        self.Base_alt = 0.0      # Alt. at base of bot.layer (not limb)
        # N_layer = 20        # Number of atm layers used in radiative transfer calculation #
        self.Layer_type = 1      # Layers weighted by mass #
        self.Layer_int = 1       # Layer intergration  scheme #

        # debug
        self.totam = None
        self.press = None
        self.temp = None
        self.delH = None
        self.scale = None

    def _name_apr(self): # this file is not used in multinest retrieval
        NVAR=1
        VARID = self.gas_id_list[0]
        f = open('{}.apr'.format(self.name),'w')
        f.write('# This is the .apr file, not used\n')
        f.write('{}\n'.format(NVAR)) # give nemesis an apr so it runs
        f.write('{:3.0f} {:3.0f} {:3.0f}\n'.format(VARID, 0, 3))
        f.write('{:6.1f} {:6.6e}\n'.format(1, 1e-6))
        f.close()

    # model atmosphere files
    # note will be modified by hydrohynamics routine
    def _name_ref(self, H_model, P_model, T_model, VMR_model, AMFORM=1,
        LATITUDE=0.0, planet_id=87):

        AMFORM = 1 # testing

        f = open('{}.ref'.format(self.name),'w')
        f.write('{}\n'.format(AMFORM))
        f.write('1\n')
        if AMFORM == 1:
            f.write('{:4} {:4} {:4} {:4}\n'.format(planet_id, LATITUDE, self.NLAYER,
                self.NVMR))
        if AMFORM == 2:
            f.write('{:4} {:4} {:4} {:4}\n'.format(planet_id, LATITUDE, self.NLAYER,
                self.NVMR))
        for i in range(len(self.gas_id_list)):
            f.write('{:4} {:4}\n'.format(self.gas_id_list[i], self.iso_id_list[i]))

        f.write('{:>14} {:>14} {:>14}'.format('height (km)', 'press (atm)', 'temp (K)'))
        for igas in range(len(self.gas_id_list)):
            f.write('{:>11} {:>2}'.format('VMR gas', igas))

        for ilayer in range(self.NLAYER):
            f.write('\n{:14.3f} {:14.5E} {:14.3f}'.format(H_model[ilayer],
                P_model[ilayer], T_model[ilayer]))
            for ivmr in range(self.NVMR):
                f.write('{:14.5E}'.format(VMR_model[ilayer,ivmr]))
        f.close()

        self.NMODEL = len(H_model)

    def write_atmospheric_files(self, H_model, P_model, T_model, VMR_model):
        """
        These files contain the state of the model atmosphere according to
        a prescribed atmosphere profile. This method should be called every time
        before the pymultinest sampler runs a forward model.
        """
        # create model atmosphere
        self._name_ref(H_model, P_model, T_model, VMR_model)
        # write input atmosphere files for the model atmosphere
        self._name_apr()

    ### useful config files
    # collision induced absorption file
    def _name_cia(self):
        f = open('{}.cia'.format(self.name),'w')
        f.write('{}\n'.format(self.cia_file_name))
        f.write('10.0\n')
        f.write('0')
        f.close()

    # Additional flags .fla file: more integer flags
    def _name_fla(self):
        f = open('{}.fla'.format(self.name),'w')
        f.write('{}\n'.format(self.INORMAL))
        f.write('{}\n'.format(self.IRAY))
        f.write('{}\n'.format(self.IH2O))
        f.write('{}\n'.format(self.ICH4))
        f.write('{}\n'.format(self.IO3))
        f.write('{}\n'.format(self.INH3))
        f.write('{}\n'.format(self.IPTF))
        f.write('{}\n'.format(self.IMIE))
        f.write('{}\n'.format(self.IUV))
        f.close()

    # Input .inp file: specific run information
    def _name_inp(self):
        f = open('{}.inp'.format(self.name),'w')
        f.write('{}  {}  {}  ! ISPACE, ISCAT, ILBL\n'.format(self.ISPACE, self.ISCAT, self.ILBL))
        f.write('{}  ! Wavenumber offset\n'.format(self.WOFF))
        f.write('{}\n'.format(self.ENAME))
        f.write('{}  ! Number of iterations, -1 for forward model\n'.format(self.NITER))
        f.write('{:.6f}  ! Minimum change in cost function\n'.format(self.PHILIMIT))
        f.write('{}      {}  ! Number of spectra to retrieve and starting ID\n'.format(self.NSPEC, self.IOFF))
        f.write('{}  ! Use previous retrieval results (set to 1 if yes)\n'.format(self.LIN))
        f.write('{}  ! unit of the calculated spectrum\n'.format(self.IFORM))
        f.close()

    # Setup .set file: scattering quadrature & layering information
    def _name_set(self,T_surface): # only the last 3 lines used
        f = open('{}.set'.format(self.name),'w')
        f.write('*'*57+'\n')
        f.write('Number of zenith angles :  {}\n'.format(5))
        f.write('  0.16527895766638701       0.32753976118389799\n')
        f.write('  0.47792494981044398       0.29204268367968400\n')
        f.write('  0.73877386510550502       0.22488934206311700\n')
        f.write('  0.91953390816645897       0.13330599085106901\n')
        f.write('  1.0000000000000000        2.2222222222222199E-002\n')
        f.write(' Number of fourier components :  {}\n'.format(1))
        f.write(' Number of azimuth angles for fourier analysis : {}\n'.format(100))
        f.write(' Sunlight on(1) or off(0) :  {}\n'.format(0))  # reflected sunlight
        f.write(' Distance from Sun (AU) :   {}\n'.format(0.015))
        f.write(' Lower boundary cond. Thermal(0) Lambert(1) :  {}\n'.format(1))
        f.write(' Ground albedo :   {:.3f}\n'.format(0.000))
        # f.write(' Surface temperature : {:.3f}\n'.format(T_surface)) #### this is trouble
        f.write(' Surface temperature : {:.3f}\n'.format(0))
        f.write('*'*57+'\n')
        f.write(' Alt. at base of bot.layer (not limb) :     {:.2f}\n'.format(self.Base_alt))
        f.write(' Number of atm layers :  {}\n'.format(self.NLAYER))
        f.write(' Layer type :  {}\n'.format(self.Layer_type))
        f.write(' Layer integration :  {}\n'.format(self.Layer_int))
        f.write('*'*57+'\n')
        f.close()

    def _name_kls(self):
        f = open('{}.kls'.format(self.name),'w')
        f.write(self.k_table_location)
        f.close()

    def _name_sol(self):
        f = open('{}.sol'.format(self.name),'w')
        f.write(self.stellar_spectrum)
        f.close()

    ### fixed config files
    # required no abort file
    def _name_abo(self):
        f = open('{}.abo'.format(self.name),'w')
        f.write('nostop')
        f.close()

    def _name_fil(self):
        f = open('{}.fil'.format(self.name),'w')
        f.write(self.filter_function)
        f.close()

    def _name_spx(self,path_angle):
        # FWHM = 0.0
        # LATITUDE = 0.0
        # LONGITUDE = 0
        # NGEOM = 1
        # NCONV = self.NWAVE
        # NAV = 1
        # FLAT = 0.0
        # FLON = 0
        # SOL_ANG = 0
        # EMISS_ANG = path_angle
        # AZI_ANG = 0
        # WEIGHT = 1.0

        # f = open('{}.spx'.format(self.name),'w')
        # f.write('{:10.5f}{:10.5f}{:10.3f}{:10}\n'.format(FWHM,LATITUDE,LONGITUDE,NGEOM))
        # f.write('{:10}\n'.format(NCONV))
        # f.write('{:10}\n'.format(NAV))
        # f.write('{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.6f}\n'.format(
        #     FLAT,FLON,SOL_ANG,EMISS_ANG,AZI_ANG,WEIGHT
        # ))
        # for iwave,wave in enumerate(self.wave_grid):
        #     f.write('{:14.6f} {:14.5E} {:14.3E}\n'.format(wave,1e-3,1e-5))
        # f.close()
        f = open('{}.spx'.format(self.name),'w')
        f.write(spx_2zenith)
        f.close()

    # run name file
    def _name_nam(self):
        f = open('{}.nam'.format(self.name),'w')
        f.write('{}'.format(self.name))
        f.close()

    # noise file
    def _noise_dat(self):
        f = open('noise.dat','w')
        f.write('2\n')
        f.write('0.1   1e-8\n')
        f.write('11. 1e-8\n')
        f.close()

    def write_config_files(self,T_surface,path_angle):
        self._noise_dat()
        self._name_abo()
        self._name_cia()
        self._name_fla()
        self._name_inp()
        self._name_kls()
        self._name_nam()
        self._name_set(T_surface)
        self._name_sol()
        self._name_fil()
        self._name_spx(path_angle)

    def _aerosol_ref(self,H_model):
        # set aerosol density to 0 for a clear atmospehre
        f = open('aerosol.ref','w')
        f.write('#aerosol.ref\n')
        f.write('{:10} {:10}\n'.format(self.NLAYER,1))
        for i in range(self.NLAYER):
            f.write('{:10.3f} {:10.3E}\n'.format(H_model[i], 0))
        f.close()

    def _name_xsc(self):
        # values don't matter since assume clear atmosphere
        f = open('{}.xsc'.format(self.name),'w')
        f.write('1\n')
        wavelen = np.linspace(0.4, 8, 20)*1e-6
        ext_xsc = np.ones(len(wavelen))
        for i in range(len(wavelen)):
            f.write('{:10.5f} {:10.5E}\n'.format(wavelen[i]*1e6, ext_xsc[i]))
            f.write('{:10.5f}\n'.format(1))
        f.close()

    def _fcloud_ref(self,H_model):
        # set fractional cloud coverage to 1 and set cloud distributino in aerosol.ref
        fcloud_N = np.ones(self.NLAYER)
        f = open('fcloud.ref', 'w')
        f.write('#fcloud.ref\n')
        f.write('{:10} {:10}\n'.format(self.NLAYER, 1))
        for i in range(self.NLAYER):
            f.write('{:10.3f} {:10.3E} {:5.0f}\n'.format(H_model[i], fcloud_N[i], 1))
        f.close()

    def write_aerosol_files(self,H_model):
        self._aerosol_ref(H_model)
        self._fcloud_ref(H_model)
        self._name_xsc()

    def write_files(self, path_angle, H_model, P_model, T_model, VMR_model):
        # get to nemesis units
        H_model = H_model * 1e-3
        P_model = P_model/const['ATM']
        self.write_config_files(T_model[0],path_angle)
        self.write_atmospheric_files(H_model, P_model, T_model, VMR_model)
        self.write_aerosol_files(H_model)

    def run_forward_model(self):
        # os.system("~/bin/Nemesis < {}.nam > /dev/null".format(self.name))
        # os.system("~/bin/Nemesis < {}.nam > stuff.out".format(self.name))
        os.system("~/bin/Nemesis < {}.nam > stuff.out".format(self.name))
        "~/bin/Nemesis < testing.nam > stuff.out"

    def read_output(self):
        wave, yerr, model = np.loadtxt("{}.mre".format(self.name), skiprows=5,
            usecols=(1,3,5), unpack=True, max_rows=self.NWAVE)
        return wave, yerr, model

    def read_prf_file(self):
        skiprows = 2 + self.NVMR + 1
        height = np.zeros(self.NMODEL)
        press = np.zeros(self.NMODEL)
        temp = np.zeros(self.NMODEL)

        iread = 0
        for imodel in range(self.NMODEL):
            skip = skiprows + iread
            height[imodel], press[imodel], temp[imodel] \
                = np.loadtxt('{}.prf'.format(self.name),skiprows=skip,
                usecols=(0,1,2), unpack=True, max_rows=1)
            iread += 1

        return height*1e3, press* const['ATM'], temp

    def read_drv_file(self):
        skiprows = 7 + 2*self.NVMR + 4
        iread = 0
        totam = np.zeros(self.NLAYER)
        pressure = np.zeros(self.NLAYER)
        temp = np.zeros(self.NLAYER)
        delH = np.zeros(self.NLAYER)
        for ilayer in range(self.NLAYER):
            skip = skiprows + iread

            delH[ilayer],totam[ilayer],pressure[ilayer],temp[ilayer] = np.loadtxt(
                '{}.drv'.format(self.name),skiprows=skip,usecols=(2,5,6,7),
                unpack=True,max_rows=1)

            iread += 4

            # line = np.loadtxt(
            #     '{}.drv'.format(self.name),skiprows=skip)
            # iread += 4
            # print(line)
        iread += 1
        scale = np.zeros(self.NLAYER)

        # note that layers are inverted
        for ilayer in range(self.NLAYER):
            skip = skiprows + iread
            ilay, jlay, Tlay, scale_lay, indent, word1, word2, word3, word4, \
                word5,word6 = np.loadtxt('{}.drv'.format(self.name),
                skiprows=skip,unpack=True,max_rows=1,dtype=str)
            scale[ilayer] = scale_lay
            iread += 1

        self.totam = totam * 1e4 # convert to number per m2
        self.pressure = pressure * const['ATM'] # convertt to Pa
        self.temp = temp
        self.delH = delH *1e3 # conver to m
        self.scale = scale[::-1]

        return delH*1e3,totam*1e4,pressure* const['ATM'],temp,scale[::-1]

"""
### Reference Planet Input
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m
R_star = 463892759.99999994 # m

### Reference Spectral Input
# Stellar spectrum
stellar_spec = np.array([3.341320e+25, 3.215455e+25, 3.101460e+25, 2.987110e+25,
       2.843440e+25, 2.738320e+25, 2.679875e+25, 2.598525e+25,
       2.505735e+25, 2.452230e+25, 2.391140e+25, 2.345905e+25,
       2.283720e+25, 2.203690e+25, 2.136015e+25, 1.234010e+24,
       4.422200e+23])

# Spectral output wavelengths in micron
wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875,
       1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6   ,
       4.5   ])

### Reference Atmospheric Model Input
# Height in m
H = np.array([      0.     ,  103738.07012,  206341.39335,  305672.8162 ,
        400037.91149,  488380.27388,  570377.57036,  646857.33242,
        718496.09845,  785987.95083,  851242.50591,  914520.46249,
        976565.39549, 1037987.38369, 1099327.5361 , 1158956.80091,
       1221026.73382, 1280661.28989, 1341043.14058, 1404762.36466])

# Pressure in pa, note 1 atm = 101325 pa
P = np.array([2.00000000e+06, 1.18757212e+06, 7.05163779e+05, 4.18716424e+05,
       2.48627977e+05, 1.47631828e+05, 8.76617219e+04, 5.20523088e+04,
       3.09079355e+04, 1.83527014e+04, 1.08975783e+04, 6.47083012e+03,
       3.84228875e+03, 2.28149750e+03, 1.35472142e+03, 8.04414702e+02,
       4.77650239e+02, 2.83622055e+02, 1.68410823e+02, 1.00000000e+02])

# Temperature in Kelvin
T = np.array([2294.22993056, 2275.69702232, 2221.47726725, 2124.54056941,
       1996.03871629, 1854.89143353, 1718.53879797, 1599.14914582,
       1502.97122783, 1431.0218576 , 1380.55933525, 1346.97814697,
       1325.49943114, 1312.13831743, 1303.97872899, 1299.05347108,
       1296.10266693, 1294.34217288, 1293.29484759, 1292.67284408])
NMODEL = len(H)
NLAYER = 20

# Ground temperature in Kelvin and path angle
T_ground = T[0]
path_angle = 0

# Gas Volume Mixing Ratio, constant with height
gas_id = np.array([ 1,  2,  5,  6, 40, 39])
iso_id = np.array([0, 0, 0, 0, 0, 0])
H2_ratio = 0.85
VMR_H2O = 1.0E-4 # volume mixing ratio of H2O
VMR_CO2 = 1.0E-4 # volume mixing ratio of CO2
VMR_CO = 1.0E-4 # volume mixing ratio of CO
VMR_CH4 = 1.0E-4 # volume mixing ratio of CH4
VMR_He = (np.ones(NMODEL)-VMR_H2O-VMR_CO2-VMR_CO-VMR_CH4)*(1-H2_ratio)
VMR_H2 = (np.ones(NMODEL)-VMR_H2O-VMR_CO2-VMR_CO-VMR_CH4)*H2_ratio
NVMR = 6
VMR = np.zeros((NMODEL,NVMR))
VMR[:,0] = VMR_H2O
VMR[:,1] = VMR_CO2
VMR[:,2] = VMR_CO
VMR[:,3] = VMR_CH4
VMR[:,4] = VMR_He
VMR[:,5] = VMR_H2


folder_name = 'testing'
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
file_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_path+'/'+folder_name) # move to designated process folder

API = Nemesis_api(name=folder_name, NLAYER=NLAYER, gas_id_list=gas_id,
    iso_id_list=iso_id,wave_grid=wave_grid)
API.write_files(path_angle=path_angle, H_model=H, P_model=P, T_model=T,
    VMR_model=VMR)
API.run_forward_model()
wave, yerr, model = API.read_output()

plt.plot(wave,model)
plt.scatter(wave,model,marker='x',color='k',linewidth=0.5,s=10,label='fortran')
plt.grid()
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.title('FORTRAN')
plt.xlabel(r'wavelength($\mu$m)')
plt.ylabel(r'total radiance(W sr$^{-1}$ $\mu$m$^{-1})$')
plt.legend()
plt.tight_layout()
plt.show()
"""
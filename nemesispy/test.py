#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from nemesispy.common.constants import R_SUN, R_JUP_E, AMU, AU, M_JUP, R_JUP

config = {
    'planet' : {
        'M_plt' : 1.898e27, # kg Jupiter mass
        'R_plt' : 6.9911e7, # m Jupiter radius
        'R_star' : 6.95700e8, # m solar radius
    },
    'gas' : {
        'gas_name_list' : np.array(['H2O','CO2','CO','He','H2']),
        'gas_id_list' : np.array([  1, 2,  5,  6, 40, 39]),
        'iso_id_list' : np.array([0, 0, 0, 0, 0, 0])
    },
    'files' : {
        'opacity' : [
            'h2owasp43.kta',
            'co2wasp43.kta',
            'cowasp43.kta',
            'ch4wasp43.kta'
        ],
        'cia' : 'exocia_hitran12_200-3800K.tab',
    },
    'settings' : {
        'nmu' : 5,
    },
    'atm' : {
        'nlayer' : 20,
        'layer_type' : 'log'
    }
}


#!/usr/bin/python3
# -*- coding: utf-8 -*-
print('loading libraries')
import numpy as np
import os
import sys
from atmosphere import unit, Model2
from utils import mol_mass, radtran_ID, phase_curve_fil, spx_2zenith
from mpi4py import MPI
import pymultinest
print('libraries loaded')

spec_1D = """0.00000      0.00000      22.5000       1
      17
      5
      0.00000      0.00000      0.00000      80.4866      0.00000     0.107272
      0.00000      0.00000      0.00000      61.4500      0.00000     0.276574
      0.00000      0.00000      0.00000      42.3729      0.00000     0.329220
      0.00000      0.00000      0.00000      23.1420      0.00000     0.242897
      0.00000      0.00000      0.00000      0.00000      0.00000    0.0440370
      1.14250  4.10000e-05  7.00000e-05
      1.17750  3.60000e-05  6.40000e-05
      1.21250  2.20000e-05  6.20000e-05
      1.24750  3.70000e-05  5.90000e-05
      1.28250  6.60000e-05  6.00000e-05
      1.31750  6.70000e-05  5.60000e-05
      1.35250  4.70000e-05  5.80000e-05
      1.38750  1.00000e-06  5.50000e-05
      1.42250  2.50000e-05  5.90000e-05
      1.45750  1.20000e-05  5.80000e-05
      1.49250  1.70000e-05  5.90000e-05
      1.52750 -5.00000e-06  5.80000e-05
      1.56250  9.00000e-06  6.10000e-05
      1.59750  3.60000e-05  6.10000e-05
      1.63250  7.00000e-05  6.60000e-05
      3.60000  8.30000e-05  0.000103000
      4.50000  0.000247000  0.000133000"""

Retrieve = True
runname = 'wasp43b'
file_path = os.path.dirname(os.path.realpath(__file__))

if Retrieve == True:
    n_live_points = 2000

def main():
    # Define Example Retrieved Parameters; prior distributions defined in Prior.
    vmr_H2O = 1.0E-20       # volume mixing ratio of H2O, constant-with-altitude and longitude
    vmr_CO2 = 1.0E-20        # volume mixing ratio of CO2, constant-with-altitude and longitude
    vmr_CO = 1.0E-20         # volume mixing ratio of CO, constant-with-altitude and longitude
    vmr_CH4 = 1.0E-10        # volume mixing ratio of CH4, constant-with-altitude and longitude

    kappa = 1e-3          # mean visible opacity
    gamma1 = 1e-1         # mean thermal opacity 1
    gamma2 = 1e-1         # mean thermal opacity 2
    alpha = 0.5           # partition of visible channels
    beta = 1000           # irradiation temperature

    free_param = {
        'vmr_H2O': vmr_H2O,
        'vmr_CO2': vmr_CO2,
        'vmr_CO': vmr_CO,
        'vmr_CH4': vmr_CH4,
        'kappa': kappa,
        'gamma1': gamma1,
        'gamma2': gamma2,
        'alpha': alpha,
        'beta': beta
    }
    n_params = len(free_param)

    print("File path:\n", file_path)
    print('Unit:\n', unit)

    if Retrieve == False:
        print('Running forward model')
        API = Nemesis_api(runname, free_param)
        API.write_config_files()
        API.write_aerosol_files()
        API.write_atmospheric_files()
        API.run_forward_model()

    else:
        print('Running retrieval mode\n')
        print('Number of parameters to retrieve: {}\n'.format(n_params))
        print('Initiating MPI')
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        print('Process rank\n', rank)
        print('Making folders')
        if rank==0:
            if not os.path.isdir("chains"):
                os.mkdir("chains")
            if not os.path.isdir(str(999)):
                os.system("mkdir "+str(999))
        if not os.path.isdir(str(rank)):
            os.system("mkdir "+str(rank))

        print('Running PyMultiNest')
        pymultinest.run(LogLikelihood = LogLikelihood,
                        Prior = Prior,
                        n_dims = n_params,
                        n_live_points = n_live_points,
                        sampling_efficiency = 0.8,
                    )
        print('Retreival Finished')

def Prior(cube, ndim, nparams):
    cube[0] = -10.0 + (-1.0+10.0)*cube[0]     # log vmr H2O
    cube[1] = -10.0 + (-1.0+10.0)*cube[1]     # log vmr CO2
    cube[2] = -10.0 + (-1.0+10.0)*cube[2]     # log vmr CO
    cube[3] = -10.0 + (-1.0+10.0)*cube[3]     # log vmr CH4

    cube[4] = -4. + (1.-(-4.))*cube[4]      # log kappa
    cube[5] = -4. + (1.-(-4.))*cube[5]      # log gamma1
    cube[6] = -4. + (1.-(-4.))*cube[6]      # log gamma2
    cube[7] = 0 + (1.-0.)*cube[7]           # alpha
    cube[8] = 0. + (3000.-0.)*cube[8]       # beta


def LogLikelihood(cube, ndim, nparams):
    # sample parameter space by drawing variables from priors
    vmr_H2O = 10.0**np.array(cube[0])
    vmr_CO2 = 10.0**np.array(cube[1])
    vmr_CO = 10.0**np.array(cube[2])
    vmr_CH4 = 10.0**np.array(cube[3])

    kappa = 10.0**np.array(cube[4])
    gamma1 = 10.0**np.array(cube[5])
    gamma2 = 10.0**np.array(cube[6])
    alpha = cube[7]
    beta = cube[8]

    free_param = {
        'vmr_H2O': vmr_H2O,
        'vmr_CO2': vmr_CO2,
        'vmr_CO': vmr_CO,
        'vmr_CH4': vmr_CH4,
        'kappa': kappa,
        'gamma1': gamma1,
        'gamma2': gamma2,
        'alpha': alpha,
        'beta': beta,
    }

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    os.chdir(file_path+'/'+str(rank)) # move to designated process folder
    API = Nemesis_api(runname, free_param) # Nemesis_api handles the forward model

    if Retrieve == False:
        API.write_config_files() # static files only need to be made once,
        API.write_aerosol_files()

    if not os.path.isfile('aerosol.ref'):
        API.write_config_files() # static files only need to be made once,
        API.write_aerosol_files()

    API.write_atmospheric_files()
    API.run_forward_model()

    # Calculate loglikelihood (chi^2); skiprows and max_rows NEED TO BE MANUALLY CHANGED
    spec, yerr, model = np.loadtxt("{}.mre".format(runname),
                                    skiprows = 5,
                                    usecols = (2,3,5),
                                    unpack = True,
                                    max_rows = 17)

    loglikelihood= -0.5*(np.sum((spec-model)**2/yerr**2))
    print(loglikelihood)

    # move back to main file folder to write to output folder (chains)
    os.chdir(file_path)

    return loglikelihood

class Nemesis_api:
    def __init__(self, runname, free_param):
        self.runname = runname
        self.atm = Nemesis_atm(runname, free_param)
        self.atm.make_atmosphere()

    def write_config_files(self):
        config = Nemesis_config(self.runname)
        config.write_config_files()

    def write_aerosol_files(self):
        aerosol = Nemesis_aerosol(self.runname, self.atm.NP, self.atm.H)
        aerosol.write_aerosol_files()

    def write_atmospheric_files(self):
        self.atm.write_atmospheric_files()

    def run_forward_model(self):
        os.system("/gf3/planetary2/PGJI011_YANG_EXOPHASE/bin/Nemesis < {}.nam > stuff.out".format(self.runname))


class Nemesis_atm:
    # static parameters, assume 85% H2 and 15% He by mole/volume
    H2_ratio = 0.85
    He_ratio = 1 - H2_ratio
    gas = np.array([
                    'H2O',
                    'CO2',
                    'CO',
                    'CH4',
                    'He',
                    'H2',
                ])
    gas_mass_amu = np.array([mol_mass(x) for x in gas]) # mmw of the gases in amu
    gas_mass_SI = gas_mass_amu*unit['amu']              # mmw of the gases in kg
    gas_ID = np.array([radtran_ID[x] for x in gas])     # radtran gas identifiers

    ID = 87                          # planet identifier; can be found in gravity.dat in raddata folder
    SMA = 0.015*unit['AU']           # Semimajor axis in m
    T_star = 4520                    # star temperature in K
    R_star = 0.6668*unit['R_sun']    # star radius in m
    M_plt = 2.052*unit['M_jup']      # planet mass in kg
    R_plt = 1.036*unit['R_jup']      # planet radius in m
    NP = 50                          # number of layers in model atmosphere
    P_top = 1e-5                     # pressure at top of model atmosphere in bar
    P_low = 2*1e1                     # pressure at bottom of model atmosphere in bar

    def __init__(self, runname, free_param):

        self.runname = runname
        # pressure range
        log_P_low = np.log10(self.P_low)
        log_P_top = np.log10(self.P_top)
        self.P_range = np.logspace(log_P_low, log_P_top, self.NP)*1e5 # pressure range in Pa

        # FREE PARAMETERS
        # retrieved; drawn from a prior distribution
        self.vmr_H2O = free_param['vmr_H2O'] # Gas VMRs
        self.vmr_CO2 = free_param['vmr_CO2']
        self.vmr_CO = free_param['vmr_CO']
        self.vmr_CH4 = free_param['vmr_CH4']

        self.kappa = free_param['kappa'] # TP profiles parameters
        self.alpha = free_param['alpha']
        self.gamma1 = free_param['gamma1']
        self.gamma2 = free_param['gamma2']
        self.beta = free_param['beta']

        # DERIVED PARAMETERS
        self.VMR_active = np.array([self.vmr_H2O, self.vmr_CO2, self.vmr_CO, self.vmr_CH4])
        VMR_He = (1-sum(self.VMR_active))*self.He_ratio
        VMR_H2 = (1-sum(self.VMR_active))*self.H2_ratio
        self.VMR = np.concatenate((self.VMR_active, np.array([VMR_He, VMR_H2]))) # list of gas VMR
        self.NVMR = len(self.VMR) # number of gases in the model atmosphere
        self.iso_id = np.zeros(self.NVMR).astype(int) # isotope identifier; assume main isotope
        self.mmw =  np.sum(self.gas_mass_SI * self.VMR) # mean molecular weight in SI unit

        # PRESSURE(atm), HEIGHT(km), TEMPERATURE(K) GRIDS
        # assign slots for atmospheric profiles in Nemesis units
        self.P = self.P_range/unit['atm'] # pressure in atm
        self.H = None
        self.T = None

    ### Method for creating a model atmosphere
    def make_atmosphere(self):
        """
        Create model atmosphere using Line et al. 2013 temperature profile.
        See batmosphere.py.
        """
        atmosphere = Model2(T_star=self.T_star, R_star=self.R_star, M_plt=self.M_plt,
                            R_plt=self.R_plt, SMA=self.SMA, P_range=self.P_range, mmw=self.mmw,
                            kappa=self.kappa, gamma1=self.gamma1,
                            gamma2=self.gamma2, alpha=self.alpha,
                            T_irr=self.beta)
        H = atmosphere.height() * 1e-3 # height in km
        T = atmosphere.temperature() # temperature in K
        self.H = H
        self.T = T

    def write_atmospheric_files(self):
        """
        These files contain the state of the model atmosphere according to
        a prescribed atmosphere profile. This method should be called every time
        before the pymultinest sampler runs a forward model.
        """
        self.make_atmosphere() # create model atmosphere
        self._name_ref() # write input atmosphere files for the model atmosphere
        self._name_apr()

    def _name_apr(self): # place_holder; give nemesis an apr so it runs
        NVAR=1
        VARID = self.gas_ID[0]
        f = open('{}.apr'.format(self.runname),'w')
        f.write('# This is the .apr file, not used\n')
        f.write('{}\n'.format(NVAR))
        f.write('{:3.0f} {:3.0f} {:3.0f}\n'.format(VARID, 0, 3))
        f.write('{:6.1f} {:6.6e}\n'.format(1, 1e-6))

    # model atmosphere files
    def _name_ref(self, AMFORM=1, LATITUDE=0.0):
        f = open('{}.ref'.format(self.runname),'w')
        f.write('{}\n'.format(AMFORM))
        f.write('1\n')
        f.write('{:4} {:4} {:4} {:4}\n'.format(self.ID, LATITUDE, self.NP, self.NVMR))
        for i in range(len(self.gas_ID)):
            f.write('{:4} {:4}\n'.format(self.gas_ID[i], self.iso_id[i]))
        f.write('  height (km)    press (atm)      temp (K)    ')
        for i in range(len(self.gas_ID)):
            f.write('VMR gas  {}   '.format(i))
        for i in range(self.NP):
            f.write('\n{:14.3f} {:14.5E} {:14.3f}'.format(self.H[i], self.P[i], self.T[i]))
            for i in range(self.NVMR):
                f.write('{:14.5E}'.format(self.VMR[i]))
        f.close()

class Nemesis_config:
    # set up .inp file
    ISPACE = 1          # 1=wavelength space (micron) #
    ISCAT = 0           # 0=thermal emission calculation #
    ILBL = 0            # 0=correlated-k
    WOFF = 0.0          # calibration error
    ENAME = 'noise.dat' # runname of the file that contains the forward modelling error #
    NITER = -1          # NUMBER OF ITERATIONS of the retrieval model required, -1 is forward model calculation #
    PHILIMIT = 0.001    # percentage convergence limit; not used in nested sampling
    NSPEC, IOFF = 1, 1  # NSPEC=no of spectra to retrieve; IOFF=index of first spectrum to fit
    LIN = 0             # 0=don't use previous retrievals to set atm prf
    IFORM = 1           # spectral unit: 0=radiance, 1=F_p/F_* (eclipse), 2=A_p/A_* (transit) #
    # set up .fla file
    INORMAL = 1         # wether ortho/para-H2 ratio is in eqlm, (0=eqm, 1=normal) #
    IRAY = 1            # Rayleigh optical depth calculation,1=gas giant,2=CO2 dominated,>2=N2,O2 dominated #
    IH2O = 0            # additional H2O continuum
    ICH4 = 0            # additional CH4 continuum
    IO3 = 0             # additional O3 continuum
    INH3 = 0            # additional NH3 continuum
    IPTF = 0            # CH4 partition function normal (0) or high-temperature (1) for Hot Jupiters.
    IMIE = 0            # phase function from hgphase*.dat, 1=from PHASEN.DAT
    IUV = 0             # additional UV opacity
    # set up .set file
    Base_alt = 0.0      # Alt. at base of bot.layer (not limb)
    N_layer = 50        # Number of atm layers used in radiative transfer calculation #
    Layer_type = 1      # Layers weighted by mass #
    Layer_int = 1       # Layer intergration  scheme #
    # set up .kls file
    k_table_location = """/gf3/planetary2/PGJI011_YANG_EXOPHASE/k_tables/h2owasp43.kta
/gf3/planetary2/PGJI011_YANG_EXOPHASE/k_tables/cowasp43.kta
/gf3/planetary2/PGJI011_YANG_EXOPHASE/k_tables/co2wasp43.kta
/gf3/planetary2/PGJI011_YANG_EXOPHASE/k_tables/ch4wasp43.kta
"""
    # set up .sol file
    stellar_spectrum = 'wasp43_stellar_newgrav.txt'
    # set up .cia file
    cia_location = 'exocia_hitran12_200-3800K.tab'
    # set up .fil file
    filter_function = phase_curve_fil
    # set up .spx file
    input_spectrum = spec_1D

    def __init__(self, runname):
        self.runname = runname

    # required no abort file
    def _name_abo(self):
        f = open('{}.abo'.format(self.runname),'w')
        f.write('nostop')
        f.close()

    # run runname file
    def _name_nam(self):
        f = open('{}.nam'.format(self.runname),'w')
        f.write('{}'.format(self.runname))
        f.close()

    # collision induced absorption file
    def _name_cia(self):
        f = open('{}.cia'.format(self.runname),'w')
        f.write('{}\n'.format(self.cia_location))
        f.write('10.0\n')
        f.write('0')
        f.close()

    # noise file
    def _noise_dat(self):
        f = open('noise.dat','w')
        f.write('2\n')
        f.write('0.1   1e-8\n')
        f.write('11. 1e-8\n')
        f.close()

    # Input .inp file: specific run information
    def _name_inp(self):
        f = open('{}.inp'.format(self.runname),'w')
        f.write('{}  {}  {}  ! ISPACE, ISCAT, ILBL\n'.format(self.ISPACE, self.ISCAT, self.ILBL))
        f.write('{}  ! Wavenumber offset\n'.format(self.WOFF))
        f.write('{}\n'.format(self.ENAME))
        f.write('{}  ! Number of iterations, -1 for forward model\n'.format(self.NITER))
        f.write('{:.6f}  ! Minimum change in cost function\n'.format(self.PHILIMIT))
        f.write('{}      {}  ! Number of spectra to retrieve and starting ID\n'.format(self.NSPEC, self.IOFF))
        f.write('{}  ! Use previous retrieval results (set to 1 if yes)\n'.format(self.LIN))
        f.write('{}  ! unit of the calculated spectrum\n'.format(self.IFORM))
        f.close()

    # Additional flags .fla file: more integer flags
    def _name_fla(self):
        f = open('{}.fla'.format(self.runname),'w')
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

    # Setup .set file: scattering quadrature & layering information
    def _name_set(self): # only the last 3 lines used
        f = open('{}.set'.format(self.runname),'w')
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
        f.write(' Surface temperature : {:.3f}\n'.format(2000))
        f.write('*'*57+'\n')
        f.write(' Alt. at base of bot.layer (not limb) :     {:.2f}\n'.format(self.Base_alt))
        f.write(' Number of atm layers :  {}\n'.format(self.N_layer))
        f.write(' Layer type :  {}\n'.format(self.Layer_type))
        f.write(' Layer integration :  {}\n'.format(self.Layer_int))
        f.write('*'*57+'\n')
        f.close()

    def _name_kls(self):
        f = open('{}.kls'.format(self.runname),'w')
        f.write(self.k_table_location)
        f.close()

    def _name_sol(self):
        f = open('{}.sol'.format(self.runname),'w')
        f.write(self.stellar_spectrum)
        f.close()

    def _name_fil(self):
        f = open('{}.fil'.format(self.runname),'w')
        f.write(self.filter_function)
        f.close()

    def _name_spx(self):
        f = open('{}.spx'.format(self.runname),'w')
        f.write(self.input_spectrum)
        f.close()

    def write_config_files(self):
        self._noise_dat()
        self._name_abo()
        self._name_cia()
        self._name_fla()
        self._name_inp()
        self._name_kls()
        self._name_nam()
        self._name_set()
        self._name_sol()
        self._name_fil()
        self._name_spx()

class Nemesis_aerosol:
    # currently not used; set to aerosol-less values
    def __init__(self, runname, NP=20, H=np.linspace(0,1000)):
        self.runname = runname
        self.NP = NP
        self.H = H

    def _aerosol_ref(self):
        # set aerosol density to 0 for a clear atmospehre
        f = open('aerosol.ref','w')
        f.write('#aerosol.ref\n')
        f.write('{:10} {:10}\n'.format(self.NP,1))
        for i in range(self.NP):
            f.write('{:10.3f} {:10.3E}\n'.format(self.H[i], 0))
        f.close()

    def _name_xsc(self):
        # values don't matter since assume clear atmosphere
        f = open('{}.xsc'.format(self.runname),'w')
        f.write('1\n')
        wavelen = np.linspace(0.4, 8, 20)*1e-6
        ext_xsc = np.ones(len(wavelen))
        for i in range(len(wavelen)):
            f.write('{:10.5f} {:10.5E}\n'.format(wavelen[i]*1e6, ext_xsc[i]))
            f.write('{:10.5f}\n'.format(1))
        f.close()

    def _fcloud_ref(self):
        # set fractional cloud coverage to 1 and set cloud distributino in aerosol.ref
        fcloud_N = np.ones(self.NP)
        f = open('fcloud.ref', 'w')
        f.write('#fcloud.ref\n')
        f.write('{:10} {:10}\n'.format(self.NP, 1))
        for i in range(self.NP):
            f.write('{:10.3f} {:10.3E} {:5.0f}\n'.format(self.H[i], fcloud_N[i], 1))
        f.close()

    def write_aerosol_files(self):
        self._aerosol_ref()
        self._fcloud_ref()
        self._name_xsc()

if __name__ == "__main__":
    main()

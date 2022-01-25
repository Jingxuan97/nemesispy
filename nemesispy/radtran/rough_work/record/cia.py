from scipy.io import FortranFile
import numpy as np
from numba import jit

NUMPAIRS = 9
NUMT = 25
NUMWN = 1501

#       Elements of absorption table are:
#       1............H2-H2 (ortho:para = 1:1 `equilibrium')
#       2............H2-He (ortho:para = 1:1 `equilibrium')
#       3............H2-H2 (ortho:para = 3:1 `normal')
#       4............H2-He (ortho:para = 3:1 `normal')
#       5............H2-N2
#       6............N2-CH4
#       7............N2-N2
#       8............CH4-CH4
#       9............H2-CH4

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def read_cia(filename,dnu=10):
    kcia_pair_t_w = np.zeros((NUMPAIRS, NUMT, NUMWN))
    nu_grid = np.linspace(0,dnu*(NUMWN-1),1501)
    f = FortranFile(filename, 'r' )
    TEMPS = f.read_reals( dtype='float64' )
    KCIA_list = f.read_reals( dtype='float32' )
    index = 0
    for iwn in range(NUMWN):
        for itemp in range(NUMT):
            for ipair in range(NUMPAIRS):
                kcia_pair_t_w[ipair,itemp,iwn] = KCIA_list[index]
                index += 1
    return TEMPS, kcia_pair_t_w, nu_grid

def interp_wn_cia(wl,kcia_pair_t_w,nu_grid):
    wl = 1e4/wl
    new_kcia_pair_t_w = np.zeros((NUMPAIRS, NUMT, len(wl)))
    for i in range(len(wl)):
        index = find_nearest(nu_grid,wl[i])
        new_kcia_pair_t_w[:,:,i] = kcia_pair_t_w[:,:,index]
    return new_kcia_pair_t_w

def interp_T_cia(T_atm,new_kcia_pair_t_w,TEMPS):
    a,b,nwave = new_kcia_pair_t_w.shape
    Nlayer = len(T_atm)
    newT_kcia = np.zeros((NUMPAIRS, len(T_atm), nwave))
    for ilayer in range(Nlayer):
        T = T_atm[ilayer]

        if T >= TEMPS[-1]:
            T = TEMPS[-1]-1
        if T <= TEMPS[0]:
            T = TEMPS[0]+1

        T_index_hi = np.where(TEMPS >= T)[0][0]
        T_index_low = np.where(TEMPS <= T)[0][-1]
        T_hi = TEMPS[T_index_hi]
        T_low = TEMPS[T_index_low]

        for ipair in range(NUMPAIRS):
            for iwave in range(nwave):
                newT_kcia[ipair, ilayer, iwave]\
                    =(T-T_low)/(T_hi-T_low)*new_kcia_pair_t_w[ipair,T_index_low,iwave]\
                    +(T_hi-T)/(T_hi-T_low)*new_kcia_pair_t_w[ipair,T_index_low,iwave]

    return newT_kcia


@jit(nopython=True)
def tau_cia(kcia_pair_l_w,U_layer,length,qh2,qhe):
    """
    kcia_pair_l_w : ndarray
        CIA absorption per pair per layer per wavelength
    length : ndarray
        path length
    """
    # need path length
    # U_layer

    U_layer = U_layer*1e20
    npair, nlayer, nwave = kcia_pair_l_w.shape

    AMAGAT = 2.68675E19

    amag1 = U_layer/length/AMAGAT
    #print('amag1',amag1)

    pre_tau = length*amag1**2
    #print('pre',pre_tau)
    #print('qh2',qh2)
    #print('qhe',qhe)

    kcia_l_w = kcia_pair_l_w[0,:,:]*qh2*qh2\
              + kcia_pair_l_w[1,:,:]*qh2*qhe

    tau_cia_w_l = np.zeros((nwave,nlayer))
    #tau_cia_l_w = kcia_w_l*pre_tau
    #print('kcia_l_w',kcia_l_w)
    for iwave in range(nwave):
        for ilayer in range(nlayer):
            tau_cia_w_l[iwave,ilayer] = kcia_l_w[ilayer,iwave]*pre_tau[ilayer]

    #print('tau',tau_cia_w_l)
    return tau_cia_w_l

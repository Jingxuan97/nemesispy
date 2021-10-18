#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from ck import read_kta, read_kls, interp_k
import matplotlib.pyplot as plt

T_set = [500,1000,1200,1400,1600,
         1800,1900,2000,2200,2400,2600,2800]
P_set = np.array([10,1e-3,])
colors=['b','r','k','g']


files = ['./data/ktables/h2o',
         './data/ktables/co2',
         './data/ktables/co',
         './data/ktables/ch4']
files = ['./data/ktables/h2o']
files = ['./data/ktables/co2']
files = ['./data/ktables/co']
files = ['./data/ktables/ch4']
kfile = files[0]

"""
T = 2500
P = 1
for count,ifile in enumerate([['./data/ktables/h2o'],['./data/ktables/co2'],
                              ['./data/ktables/co'],['./data/ktables/ch4']][3:4]):
    gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
            k_gas_w_g_p_t = read_kls(ifile)
    P_layer = np.array([P])
    T_layer = np.array([T])
    k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
    Nwave = len(wave_grid)
    k_plot = np.zeros(len(wave_grid))
    for i in range(Nwave):
        k_plot[i] = np.sum(del_g*k_gas_w_g_l[0,i,:,0])
    plt.scatter(wave_grid,k_plot,label=ifile[0][-3:],color=colors[count],marker='x',linewidth=0.8)
print(k_plot)
#plt.legend()
plt.yscale('log')
plt.xlim(1,5)
#plt.ylim(1e-30,1e1)


files = ['./data/ktables/H2O_Katy_ARIEL_test',
          './data/ktables/CO2_Katy_ARIEL_test',
          './data/ktables/CO_Katy_ARIEL_test',
          './data/ktables/CH4_Katy_ARIEL_test']
files = ['./data/ktables/H2O_Katy_ARIEL_test']
files = ['./data/ktables/CO2_Katy_ARIEL_test']
files = ['./data/ktables/CO_Katy_ARIEL_test']
files = ['./data/ktables/CH4_Katy_ARIEL_test']
kfile = files[0]
for count,ifile in enumerate([['./data/ktables/H2O_Katy_ARIEL_test'],
                              ['./data/ktables/CO2_Katy_ARIEL_test'],
                              ['./data/ktables/CO_Katy_ARIEL_test'],
                              ['./data/ktables/CH4_Katy_ARIEL_test']][3:4]):
    gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
            k_gas_w_g_p_t = read_kls(ifile)
    P_layer = np.array([P])
    T_layer = np.array([T])
    k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
    Nwave = len(wave_grid)
    k_plot = np.zeros(len(wave_grid))
    for i in range(Nwave):
        k_plot[i] = np.sum(del_g*k_gas_w_g_l[0,i,:,0])
    plt.plot(wave_grid,k_plot,label=ifile[0][14:18],color=colors[count],linestyle='--')
print(k_plot)

files = ['./data/ktables/H2O_Katy_R1000',
         './data/ktables/CO2_Katy_R1000',
          './data/ktables/CO_Katy_R1000',
          './data/ktables/CH4_Katy_R1000']
files = ['./data/ktables/H2O_Katy_R1000']
files = ['./data/ktables/CO2_Katy_R1000']
files = ['./data/ktables/CO_Katy_R1000']
files = ['./data/ktables/CH4_Katy_R1000']
kfile = files[0]
for count,ifile in enumerate([['./data/ktables/H2O_Katy_R1000'],
                              ['./data/ktables/CO2_Katy_R1000'],
                              ['./data/ktables/CO_Katy_R1000'],
                              ['./data/ktables/CH4_Katy_R1000']][3:4]):
    gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
            k_gas_w_g_p_t = read_kls(ifile)
    P_layer = np.array([P])
    T_layer = np.array([T])
    k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
    Nwave = len(wave_grid)
    k_plot = np.zeros(len(wave_grid))
    for i in range(Nwave):
        k_plot[i] = np.sum(del_g*k_gas_w_g_l[0,i,:,0])
    plt.plot(wave_grid,k_plot,label=ifile[0][14:18],linewidth=0.1,color=colors[count],)
print(k_plot)
#plt.legend()
plt.yscale('log')
plt.xlim(1,5)
plt.ylim(1e-20,1e2)
plt.ylabel('k x 1e20 (cm2/particle)')
plt.xlabel('wavelength (um')
plt.title('CH4 ktable T={}K, P={}bar'.format(T,P))
plt.savefig('ch4_k_T{}_P{}.pdf'.format(T,P),dpi=400)
#plt.close()
"""

for P in P_set:
    for T in T_set:
        #T = 2500
        #P = 1
        for count,ifile in enumerate([['./data/ktables/h2o'],['./data/ktables/co2'],
                                      ['./data/ktables/co'],['./data/ktables/ch4']][3:4]):
            gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
                    k_gas_w_g_p_t = read_kls(ifile)
            P_layer = np.array([P])
            T_layer = np.array([T])
            k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
            Nwave = len(wave_grid)
            k_plot = np.zeros(len(wave_grid))
            for i in range(Nwave):
                k_plot[i] = np.sum(del_g*k_gas_w_g_l[0,i,:,0])
            plt.scatter(wave_grid,k_plot,label=ifile[0][-3:],color=colors[count],marker='x',linewidth=0.8)
        print(k_plot)
        #plt.legend()
        plt.yscale('log')
        plt.xlim(1,5)
        #plt.ylim(1e-30,1e1)
        
        
        
        
        files = ['./data/ktables/H2O_Katy_ARIEL_test',
                  './data/ktables/CO2_Katy_ARIEL_test',
                  './data/ktables/CO_Katy_ARIEL_test',
                  './data/ktables/CH4_Katy_ARIEL_test']
        files = ['./data/ktables/H2O_Katy_ARIEL_test']
        files = ['./data/ktables/CO2_Katy_ARIEL_test']
        files = ['./data/ktables/CO_Katy_ARIEL_test']
        files = ['./data/ktables/CH4_Katy_ARIEL_test']
        kfile = files[0]
        for count,ifile in enumerate([['./data/ktables/H2O_Katy_ARIEL_test'],
                                      ['./data/ktables/CO2_Katy_ARIEL_test'],
                                      ['./data/ktables/CO_Katy_ARIEL_test'],
                                      ['./data/ktables/CH4_Katy_ARIEL_test']][3:4]):
            gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
                    k_gas_w_g_p_t = read_kls(ifile)
            P_layer = np.array([P])
            T_layer = np.array([T])
            k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
            Nwave = len(wave_grid)
            k_plot = np.zeros(len(wave_grid))
            for i in range(Nwave):
                k_plot[i] = np.sum(del_g*k_gas_w_g_l[0,i,:,0])
            plt.plot(wave_grid,k_plot,label=ifile[0][14:18],color=colors[count],linestyle='--')
        print(k_plot)
        
        files = ['./data/ktables/H2O_Katy_R1000',
                 './data/ktables/CO2_Katy_R1000',
                  './data/ktables/CO_Katy_R1000',
                  './data/ktables/CH4_Katy_R1000']
        files = ['./data/ktables/H2O_Katy_R1000']
        files = ['./data/ktables/CO2_Katy_R1000']
        files = ['./data/ktables/CO_Katy_R1000']
        files = ['./data/ktables/CH4_Katy_R1000']
        kfile = files[0]
        for count,ifile in enumerate([['./data/ktables/H2O_Katy_R1000'],
                                      ['./data/ktables/CO2_Katy_R1000'],
                                      ['./data/ktables/CO_Katy_R1000'],
                                      ['./data/ktables/CH4_Katy_R1000']][3:4]):
            gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
                    k_gas_w_g_p_t = read_kls(ifile)
            P_layer = np.array([P])
            T_layer = np.array([T])
            k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
            Nwave = len(wave_grid)
            k_plot = np.zeros(len(wave_grid))
            for i in range(Nwave):
                k_plot[i] = np.sum(del_g*k_gas_w_g_l[0,i,:,0])
            plt.plot(wave_grid,k_plot,label=ifile[0][14:18],linewidth=0.1,color=colors[count],)
        print(k_plot)
        #plt.legend()
        plt.yscale('log')
        plt.xlim(1,5)
        plt.ylim(1e-6,1e2)
        plt.ylabel('k x 1e20 (cm2/particle)')
        plt.xlabel('wavelength (um')
        plt.title('CH4 ktable T={}K, P={}bar'.format(T,P))
        plt.savefig('ch4_k_T{}_P{}.pdf'.format(T,P),dpi=400)
        plt.close()


class TestReadkta(unittest.TestCase):
    
    def test_read_one_table(self):
        gas_id, iso_id, wave_grid, g_ord, del_g, P_grid, T_grid, k_w_g_p_t \
            = read_kta(kfile)
        self.assertTrue(type(gas_id)==int)
        self.assertTrue(type(iso_id)==int)
        Nwave = len(wave_grid)
        Ng = len(g_ord)
        Npress = len(P_grid)
        Ntemp = len(T_grid)
        self.assertEqual(Ng, len(del_g))
        self.assertEqual(k_w_g_p_t.shape, (Nwave,Ng,Npress,Ntemp))
        
###############################################################################
def test_read_kls1():
    files = ['./data/ktables/h2o',
             './data/ktables/co2',
             './data/ktables/co',
             './data/ktables/ch4']

    gas_id_list, iso_id_list, wave_grid, g_ord, del_g,\
        P_grid, T_grid, k_gas_w_g_p_t = read_kls(files)

    print('gas_id_list',gas_id_list)
    print('iso_id_list',iso_id_list)
    print('wave_grid',wave_grid)
    print('Nwave',len(wave_grid))
    print('g_ord',g_ord)
    print('Ng',len(g_ord))
    print('del_g',del_g)
    print('Ng',len(del_g))
    print('P_grid',P_grid)
    print('Npress',len(P_grid))
    print('T_grid',T_grid)
    print('Ntemp',len(T_grid))
    print('k-shape',k_gas_w_g_p_t.shape)
# test_read_kls1()

def test_read_kls2():
    files = ['./data/ktables/H2O_Katy_R1000',
    './data/ktables/CO2_Katy_R1000',
    './data/ktables/CO_Katy_R1000',
    './data/ktables/CH4_Katy_R1000']

    gas_id_list, iso_id_list, wave_grid, g_ord, del_g,\
        P_grid, T_grid, k_gas_w_g_p_t = read_kls(files)

    print('gas_id_list',gas_id_list)
    print('iso_id_list',iso_id_list)
    print('wave_grid',wave_grid)
    print('Nwave',len(wave_grid))
    print('g_ord',g_ord)
    print('Ng',len(g_ord))
    print('del_g',del_g)
    print('Ng',len(del_g))
    print('P_grid',P_grid)
    print('Npress',len(P_grid))
    print('T_grid',T_grid)
    print('Ntemp',len(T_grid))
    print('k-shape',k_gas_w_g_p_t.shape)
# test_read_kls2()

def test_interp_k():
    print('test interp k')

    files = ['./data/ktables/h2o','./data/ktables/co2','./data/ktables/co','./data/ktables/ch4']
    gas_id_list, iso_id_list, wave, g_ord, del_g,\
        P_grid, T_grid, k_gas_w_g_p_t = read_kls(files)
    P_layer = np.array([7.42239701e+05, 2.81660408e+05, 1.08371277e+05, 4.22976280e+04,
          1.67426473e+04, 6.66814550e+03, 2.69006862e+03, 1.10450126e+03,
          4.54217702e+02, 1.86590540e+02])*1e-5
    T_layer = np.array([3014.12747492, 3014.11537511, 3014.11076141, 3014.30110731,
          3036.04559052, 3204.23046439, 3739.65219224, 4334.23131605,
          4772.68340169, 4964.2941274 ])*0.5
    """
    P_grid = np.log10(P_grid)
    T_grid = np.log10(T_grid)
    P_layer = np.log10(P_layer)
    T_layer = np.log10(T_layer)
    k_gas_w_g_p_t = np.log10(k_gas_w_g_p_t)
    """
    for i in range(100):
        k_out = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)


    print('k shape', k_out.shape)
# test_interp_k()
"""
def test_mix_two_gas_k():
    files = ['./data/ktables/h2o','./data/ktables/co2','./data/ktables/co','./data/ktables/ch4']
    gas_id_list, iso_id_list, wave, g_ord, del_g,\
        P_grid, T_grid, k_gas_w_g_p_t = read_kls(files)
    P_layer = np.array([7.42239701e+05, 2.81660408e+05, 1.08371277e+05, 4.22976280e+04,
          1.67426473e+04, 6.66814550e+03, 2.69006862e+03, 1.10450126e+03,
          4.54217702e+02, 1.86590540e+02])*1e-5
    T_layer = np.array([3014.12747492, 3014.11537511, 3014.11076141, 3014.30110731,
          3036.04559052, 3204.23046439, 3739.65219224, 4334.23131605,
          4772.68340169, 4964.2941274 ])*0.5
    frac = np.array([[1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10]])
    k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
    k_g1 = k_gas_w_g_l[0,1,:,0]
    k_g2 = k_gas_w_g_l[1,1,:,0]
    VMR1 = frac[0,1]
    VMR2 = frac[0,2]

    for i in range(100):
        k_mix, vmr_mix = mix_two_gas_k(k_g1,k_g2,VMR1,VMR2,g_ord,del_g)

    print('k_mix',k_mix)
    print('time',e-s)
# test_mix_two_gas_k()

def test_mix_multi_gas_k():
    files = ['./data/ktables/h2o','./data/ktables/co2','./data/ktables/co','./data/ktables/ch4']
    gas_id_list, iso_id_list, wave, g_ord, del_g,\
        P_grid, T_grid, k_gas_w_g_p_t = read_kls(files)
    P_layer = np.array([7.42239701e+05, 2.81660408e+05, 1.08371277e+05, 4.22976280e+04,
          1.67426473e+04, 6.66814550e+03, 2.69006862e+03, 1.10450126e+03,
          4.54217702e+02, 1.86590540e+02])*1e-5
    T_layer = np.array([3014.12747492, 3014.11537511, 3014.11076141, 3014.30110731,
          3036.04559052, 3204.23046439, 3739.65219224, 4334.23131605,
          4772.68340169, 4964.2941274 ])*0.5
    frac = np.array([[1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10]])
    k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
    k = k_gas_w_g_l[:,1,:,0]
    VMR = frac[0,:]

    for i in range(100):
        k_mix, vmr_mix = mix_multi_gas_k(k,VMR,g_ord,del_g)

    print('k_mix',k_mix)
    print('time',e-s)
# test_mix_multi_gas_k()

def test_tau_k():
    files = ['./data/ktables/h2o','./data/ktables/co2','./data/ktables/co','./data/ktables/ch4']
    gas_id_list, iso_id_list, wave, g_ord, del_g,\
        P_grid, T_grid, k_gas_w_g_p_t = read_kls(files)
    P_layer = np.array([7.42239701e+05, 2.81660408e+05, 1.08371277e+05, 4.22976280e+04,
          1.67426473e+04, 6.66814550e+03, 2.69006862e+03, 1.10450126e+03,
          4.54217702e+02, 1.86590540e+02])*1e-5
    T_layer = np.array([3014.12747492, 3014.11537511, 3014.11076141, 3014.30110731,
          3036.04559052, 3204.23046439, 3739.65219224, 4334.23131605,
          4772.68340169, 4964.2941274 ])*0.5
    totam = np.array([7.86544802e+30, 3.01842190e+30, 1.16818592e+30, 4.56144181e+29,
                    1.79953458e+29, 7.23170929e+28, 2.84773251e+28, 1.16671235e+28,
                    4.88662629e+27, 2.09772744e+27])*1e-4*1e-20
    VMR = np.array([[1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10]])

    for i in range(10000):
        k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
        k_gas_g_l = k_gas_w_g_l[:,0,:,:]
        tau = tau_k(k_gas_g_l,totam,VMR,g_ord,del_g)

    print('time',e-s)
# test_tau_k()

def test_all():
    files = ['./data/ktables/h2o','./data/ktables/co2','./data/ktables/co','./data/ktables/ch4']
    gas_id_list, iso_id_list, wave, g_ord, del_g,\
        P_grid, T_grid, k_gas_w_g_p_t = read_kls(files)
    P_layer = np.array([7.42239701e+05, 2.81660408e+05, 1.08371277e+05, 4.22976280e+04,
          1.67426473e+04, 6.66814550e+03, 2.69006862e+03, 1.10450126e+03,
          4.54217702e+02, 1.86590540e+02])*1e-5
    T_layer = np.array([3014.12747492, 3014.11537511, 3014.11076141, 3014.30110731,
          3036.04559052, 3204.23046439, 3739.65219224, 4334.23131605,
          4772.68340169, 4964.2941274 ])*0.5
    frac = np.array([[1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10]])

    for i in range(1000000):
        k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
        k = k_gas_w_g_l[:,1,:,0]
        VMR = frac[0,:]
        k_mix, vmr_mix = mix_multi_gas_k(k,VMR,g_ord,del_g)

    print('k_mix',k_mix)
    print('time',e-s)
# test_all()

def test_radiance_lowres():
    files = ['./data/ktables/h2o','./data/ktables/co2','./data/ktables/co','./data/ktables/ch4']
    gas_id_list, iso_id_list, wave, g_ord, del_g,\
        P_grid, T_grid, k_gas_w_g_p_t = read_kls(files)
    P_layer = np.array([7.42239701e+05, 2.81660408e+05, 1.08371277e+05, 4.22976280e+04,
          1.67426473e+04, 6.66814550e+03, 2.69006862e+03, 1.10450126e+03,
          4.54217702e+02, 1.86590540e+02])*1e-5
    T_layer = np.array([2900, 2500, 2200, 2000,1800.,1500,1400,1300,1200,1100])*0.1
    totam = np.array([7.86544802e+30, 3.01842190e+30, 1.16818592e+30, 4.56144181e+29,
                    1.79953458e+29, 7.23170929e+28, 2.84773251e+28, 1.16671235e+28,
                    4.88662629e+27, 2.09772744e+27])*1e-4*1e-20
    wasp = np.array([3.34097e+25,3.24149e+25,3.10545e+25,2.99568e+25,2.82785e+25,
        2.73007e+25,2.69301e+25,2.59410e+25,2.47917e+25,2.45594e+25,2.31012e+25,
        2.34431e+25,2.29190e+25,2.20532e+25,2.15548e+25,1.23401e+24,4.42220e+23])
    VMR = np.array([[1.e-4, 1.e-10, 1.e-10, 1.e-10],
                    [1.e-4, 1.e-10, 1.e-10, 1.e-10],
                    [1.e-4, 1.e-10, 1.e-10, 1.e-10],
                    [1.e-4, 1.e-10, 1.e-10, 1.e-10],
                    [1.e-4, 1.e-10, 1.e-10, 1.e-10],
                    [1.e-4, 1.e-10, 1.e-10, 1.e-10],
                    [1.e-4, 1.e-10, 1.e-10, 1.e-10],
                    [1.e-4, 1.e-10, 1.e-10, 1.e-10],
                    [1.e-4, 1.e-10, 1.e-10, 1.e-10],
                    [1.e-4, 1.e-10, 1.e-10, 1.e-10]])

    for i in range(100):
        r = radiance(wave, totam, P_layer, T_layer, VMR, k_gas_w_g_p_t,
                     P_grid, T_grid, g_ord, del_g)


    import matplotlib.pyplot as plt
    plt.scatter(wave,r, s=10, marker='x')
    plt.grid()
    #plt.xscale('log')
    #plt.xlim(1,1.6)
    #plt.ylim(0,1.4e-22)
    plt.xlabel('wavelength(um)')
    plt.ylabel('radiance')

    print('time',e-s)
# test_radiance_lowres()

def test_radiance_hires():
    files = ['./data/ktables/H2O_Katy_R1000',
    './data/ktables/CO2_Katy_R1000',
    './data/ktables/CO_Katy_R1000',
    './data/ktables/CH4_Katy_R1000']
    gas_id_list, iso_id_list, wave, g_ord, del_g,\
        P_grid, T_grid, k_gas_w_g_p_t = read_kls(files)
    P_layer = np.array([7.42239701e+05, 2.81660408e+05, 1.08371277e+05, 4.22976280e+04,
          1.67426473e+04, 6.66814550e+03, 2.69006862e+03, 1.10450126e+03,
          4.54217702e+02, 1.86590540e+02])*1e-5
    T_layer = np.array([2900, 2500, 2200, 2000,1800.,1500,1400,1300,1200,1100])
    totam = np.array([7.86544802e+30, 3.01842190e+30, 1.16818592e+30, 4.56144181e+29,
                    1.79953458e+29, 7.23170929e+28, 2.84773251e+28, 1.16671235e+28,
                    4.88662629e+27, 2.09772744e+27])*1e-4*1e-20
    wasp = np.array([3.34097e+25,3.24149e+25,3.10545e+25,2.99568e+25,2.82785e+25,
        2.73007e+25,2.69301e+25,2.59410e+25,2.47917e+25,2.45594e+25,2.31012e+25,
        2.34431e+25,2.29190e+25,2.20532e+25,2.15548e+25,1.23401e+24,4.42220e+23])
    VMR = np.array([[1.e-4, 1.e-10, 1.e-10, 1.e-10],
                    [1.e-4, 1.e-10, 1.e-10, 1.e-10],
                    [1.e-4, 1.e-10, 1.e-10, 1.e-10],
                    [1.e-4, 1.e-10, 1.e-10, 1.e-10],
                    [1.e-4, 1.e-10, 1.e-10, 1.e-10],
                    [1.e-4, 1.e-10, 1.e-10, 1.e-10],
                    [1.e-4, 1.e-10, 1.e-10, 1.e-10],
                    [1.e-4, 1.e-10, 1.e-10, 1.e-10],
                    [1.e-4, 1.e-10, 1.e-10, 1.e-10],
                    [1.e-4, 1.e-10, 1.e-10, 1.e-10]])

    for i in range(1):
        r = radiance(wave, totam, P_layer, T_layer, VMR, k_gas_w_g_p_t,
                     P_grid, T_grid, g_ord, del_g)

    print('time',e-s)
    import matplotlib.pyplot as plt
    plt.scatter(wave,r, s=10, marker='x')
    plt.grid()
    #plt.xscale('log')
    #plt.xlim(1,1.6)
    #plt.ylim(0,1.4e-22)
    plt.xlabel('wavelength(um)')
    plt.ylabel('radiance')

    print('time',e-s)
"""
def test():
    files = ['./data/ktables/h2o','./data/ktables/co2','./data/ktables/co','./data/ktables/ch4']
    files = ['./data/ktables/H2O_Katy_R1000',
    './data/ktables/CO2_Katy_R1000',
    './data/ktables/CO_Katy_R1000',
    './data/ktables/CH4_Katy_R1000']
    gas_id_list, iso_id_list, wave, g_ord, del_g,\
        P_grid, T_grid, k_gas_w_g_p_t = read_kls(files)
    P_layer = np.array([7.42239701e+05, 2.81660408e+05, 1.08371277e+05, 4.22976280e+04,
          1.67426473e+04, 6.66814550e+03, 2.69006862e+03, 1.10450126e+03,
          4.54217702e+02, 1.86590540e+02])*1e-5
    T_layer = np.array([2900, 2500, 2200, 2000,1800.,1500,1400,1300,1200,1100])*0.5
    totam = np.array([7.86544802e+30, 3.01842190e+30, 1.16818592e+30, 4.56144181e+29,
                    1.79953458e+29, 7.23170929e+28, 2.84773251e+28, 1.16671235e+28,
                    4.88662629e+27, 2.09772744e+27])*1e-4*1e-20
    wasp = np.array([3.34097e+25,3.24149e+25,3.10545e+25,2.99568e+25,2.82785e+25,
        2.73007e+25,2.69301e+25,2.59410e+25,2.47917e+25,2.45594e+25,2.31012e+25,
        2.34431e+25,2.29190e+25,2.20532e+25,2.15548e+25,1.23401e+24,4.42220e+23])
    VMR = np.array([[1.e-8, 1.e-8, 1.e-8, 1.e-8],
                    [1.e-8, 1.e-8, 1.e-8, 1.e-8],
                    [1.e-8, 1.e-8, 1.e-8, 1.e-8],
                    [1.e-8, 1.e-8, 1.e-8, 1.e-8],
                    [1.e-8, 1.e-8, 1.e-8, 1.e-8],
                    [1.e-8, 1.e-8, 1.e-8, 1.e-8],
                    [1.e-8, 1.e-8, 1.e-8, 1.e-8],
                    [1.e-8, 1.e-8, 1.e-8, 1.e-8],
                    [1.e-8, 1.e-8, 1.e-8, 1.e-8],
                    [1.e-8, 1.e-8, 1.e-8, 1.e-8]])
    s = time.time()
    for i in range(1):
        r = radiance(wave, totam, P_layer, T_layer, VMR, k_gas_w_g_p_t,
                     P_grid, T_grid, g_ord, del_g)
    e = time.time()
    print('time',e-s)

    import matplotlib.pyplot as plt
    #plt.scatter(wave,r, s=10, marker='x')
    #plt.grid()
    #plt.xscale('log')
    #plt.xlim(1,1.6)
    #plt.ylim(0,1.4e-22)
    #plt.xlabel('wavelength(um)')
    #plt.ylabel('radiance')

    plt.plot(wave,r,linewidth=0.1,color='k')
    plt.grid()
    #plt.xscale('log')
    plt.xlim(1,10)
    #plt.ylim(0,1.4e-22)
    plt.xlabel(r'wavelength($\mu$m)')
    plt.ylabel(r'radiance(W sr$^{-1}$ cm$^{-2}$ $\mu$m$^{-1}$')
    plt.tight_layout()
    plt.savefig('test_bb.pdf',dpi=400)
# test()

"""Tests"""
# test_radiance_hires()
if __name__ == '__main__':
    unittest.main()
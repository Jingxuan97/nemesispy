#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from ck import read_kta, read_kls, interp_k, mix_two_gas_k
import matplotlib.pyplot as plt

lowres_files = ['./data/ktables/h2o',
         './data/ktables/co2',
         './data/ktables/co',
         './data/ktables/ch4']

aeriel_files = ['./data/ktables/H2O_Katy_ARIEL_test',
          './data/ktables/CO2_Katy_ARIEL_test',
          './data/ktables/CO_Katy_ARIEL_test',
          './data/ktables/CH4_Katy_ARIEL_test']

hires_files = ['./data/ktables/H2O_Katy_R1000',
         './data/ktables/CO2_Katy_R1000',
          './data/ktables/CO_Katy_R1000',
          './data/ktables/CH4_Katy_R1000']

"""
import time
s = time.time()
gas_id_list, iso_id_list, wave_grid, g_ord, del_g,\
P_grid, T_grid, k_gas_w_g_p_t = read_kls(aeriel_files)
e = time.time()
print('reading time',e-s)
s = time.time()
for i in range(1000):
    Nlay = 100
    P_layer = np.logspace(1,-3,Nlay)
    T_layer = np.linspace(2000,500,Nlay)
    k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
e = time.time()
print('interp time',e-s)
"""

gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
    k_gas_w_g_p_t = read_kls(aeriel_files)
P_layer = np.array([1])
T_layer = np.array([2000])
Nwave = len(wave_grid)
k_plot = np.zeros(len(wave_grid))
k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
VMR1 = 0.5
VMR2 = 0.5
k_g1 = k_gas_w_g_l[1,10,:,0]
k_g2 = k_gas_w_g_l[0,10,:,0]
k_g_mix, vmr_mix = mix_two_gas_k(k_g1, k_g2, VMR1, VMR2, g_ord, del_g)
plt.scatter(g_ord,k_g1,label='k_g1',marker='x')
plt.scatter(g_ord,k_g2,label='k_g2',marker='o',s=10)
plt.plot(g_ord,k_g_mix,label='k_g_mix',color='k')
plt.yscale('log')
plt.legend()

"""
class TestReadkTables(unittest.TestCase):
    
    def test_read_one_table(self):
        # check k table dimensions
        for kfile in lowres_files+aeriel_files+hires_files:
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
            
    def test_read_multiple_tables(self):
        # check multiple k table reading
        for files in [lowres_files,aeriel_files,hires_files]:
            gas_id_list, iso_id_list, wave_grid, g_ord, del_g,\
            P_grid, T_grid, k_gas_w_g_p_t = read_kls(files)
            Ngas = len(gas_id_list)
            Nwave = len(wave_grid)
            Ng = len(g_ord)
            Npress = len(P_grid)
            Ntemp = len(T_grid)
            self.assertEqual(k_gas_w_g_p_t.shape, (Ngas,Nwave,Ng,Npress,Ntemp))

class TestInterpkTables(unittest.TestCase):
    
    def test_interp_one_gas(self):
        gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
            k_gas_w_g_p_t = read_kls(hires_files[:1])
        Nwave = len(wave_grid)
        P_layer = np.array([1])
        T_layer = np.array([1000])
        k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
        k_w = np.zeros(len(wave_grid))
        for i in range(Nwave):
            k_w[i] = np.sum(del_g*k_gas_w_g_l[0,i,:,0])
        
"""        
"""
def plot_k(files,P,T,legends,name=None):
    for index, table in enumerate(files):
        print(table)
        gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
            k_gas_w_g_p_t = read_kls(table)
        Nwave = len(wave_grid)
        k_plot = np.zeros(len(wave_grid))
        k_gas_w_g_l = interp_k(P_grid, T_grid, P, T, k_gas_w_g_p_t)
        for i in range(Nwave):
            k_plot[i] = np.sum(del_g*k_gas_w_g_l[0,i,:,0])   
        plt.plot(wave_grid,k_plot*1e-20,linewidth=0.5,label=legends[index])
        plt.legend(loc='lower right')
        plt.yscale('log')
        plt.tight_layout()
        plt.ylabel('k (cm$^2$/particle)')
        plt.xlabel('wavelength (micron)')
        if name:
            plt.savefig(name,dpi=400)
            
files = [['./data/ktables/H2O_Katy_R1000'],
         ['./data/ktables/CO2_Katy_R1000'],
         ['./data/ktables/CO_Katy_R1000'],
         ['./data/ktables/CH4_Katy_R1000']]
P = np.array([1])
T = np.array([2000])
legends = ['H2O','CO2','CO','CH4']
plot_k(files,P,T,legends)
"""


"""    
gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
    k_gas_w_g_p_t = read_kls(hires_files[0:1])
P_layer = np.array([1])
T_layer = np.array([1000])
Nwave = len(wave_grid)
k_plot = np.zeros(len(wave_grid))
k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
for i in range(Nwave):
    k_plot[i] = np.sum(del_g*k_gas_w_g_l[0,i,:,0])
plt.plot(wave_grid,k_plot,linewidth=0.5,color='b',label='H2O')


gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
    k_gas_w_g_p_t = read_kls(hires_files[1:2])
P_layer = np.array([1])
T_layer = np.array([1000])
Nwave = len(wave_grid)
k_plot = np.zeros(len(wave_grid))
k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
for i in range(Nwave):
    k_plot[i] = np.sum(del_g*k_gas_w_g_l[0,i,:,0])
plt.plot(wave_grid,k_plot,linewidth=0.5,color='r', label='CO2')


gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
    k_gas_w_g_p_t = read_kls(hires_files[2:3])
P_layer = np.array([1])
T_layer = np.array([1000])
Nwave = len(wave_grid)
k_plot = np.zeros(len(wave_grid))
k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
for i in range(Nwave):
    k_plot[i] = np.sum(del_g*k_gas_w_g_l[0,i,:,0])
plt.plot(wave_grid,k_plot,linewidth=0.5,color='k', label='CO')


gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
    k_gas_w_g_p_t = read_kls(hires_files[3:4])
P_layer = np.array([1])
T_layer = np.array([1000])
Nwave = len(wave_grid)
k_plot = np.zeros(len(wave_grid))
k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
for i in range(Nwave):
    k_plot[i] = np.sum(del_g*k_gas_w_g_l[0,i,:,0])
plt.plot(wave_grid,k_plot,linewidth=0.5,color='orange',label='CH4')
"""

"""
gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
    k_gas_w_g_p_t = read_kls(hires_files[3:4])
P_layer = np.array([1])
T_layer = np.array([1000])
Nwave = len(wave_grid)
k_plot = np.zeros(len(wave_grid))
k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
for i in range(Nwave):
    k_plot[i] = np.sum(del_g*k_gas_w_g_l[0,i,:,0])
plt.plot(wave_grid,k_plot,linewidth=0.5,color='b')
"""

"""
plt.yscale('log')
plt.xlim(0,10)
plt.ylim(1e-36,1e2)
plt.title('T=1000k, P=1bar')
plt.ylabel('k x 1e20 (cm$^2$/particle)')
plt.xlabel('wavelength (micron)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('T1000KP1bar.pdf',dpi=400)
"""

"""
if __name__ == '__main__':
    unittest.main()
"""
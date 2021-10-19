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

if __name__ == '__main__':
    unittest.main()
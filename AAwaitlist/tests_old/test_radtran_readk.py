#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from pathlib import Path
from nemesispy.radtran.readk import read_kta, read_kls


# print(lowres_files)
"""
aeriel_files = ['{}/data/ktables/H2O_Katy_ARIEL_test'.format(nemesispy_path),
          '{}/data/ktables/CO2_Katy_ARIEL_test'.format(nemesispy_path),
          '{}/data/ktables/CO_Katy_ARIEL_test'.format(nemesispy_path),
          '{}/data/ktables/CH4_Katy_ARIEL_test'.format(nemesispy_path)]

hires_files = ['{}/data/ktables/H2O_Katy_R1000'.format(nemesispy_path),
         '{}/data/ktables/CO2_Katy_R1000'.format(nemesispy_path),
          '{}/data/ktables/CO_Katy_R1000'.format(nemesispy_path),
          '{}/data/ktables/CH4_Katy_R1000'.format(nemesispy_path)]
"""

class TestReadk(unittest.TestCase):

    # find path to example low resolution k tables
    nemesispy_path = Path(__file__).parents[1]
    h2o_lowres_table = '{}/data/ktables/h2o'.format(nemesispy_path)
    co2_lowres_table = '{}/data/ktables/co2'.format(nemesispy_path)
    co_lowres_table = '{}/data/ktables/co'.format(nemesispy_path)
    ch4_lowres_table = '{}/data/ktables/ch4'.format(nemesispy_path)
    lowres_files = [h2o_lowres_table, co2_lowres_table, co_lowres_table,
        ch4_lowres_table]

    wave_grid_lowres = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825,
    1.3175, 1.3525, 1.3875, 1.4225, 1.4575, 1.4925, 1.5275, 1.5625,
    1.5975, 1.6325, 3.6, 4.5], dtype="float32")

    g_ord_lowres = np.array([0.0034357, 0.01801404, 0.04388279, 0.08044151,
    0.12683405, 0.18197316, 0.2445665, 0.31314695, 0.3861071, 0.46173674,
    0.53826326, 0.6138929, 0.68685305, 0.7554335, 0.81802684, 0.87316597,
    0.91955847, 0.9561172, 0.981986, 0.9965643 ], dtype="float32")

    del_g_lowres = np.array([0.008807, 0.02030071, 0.03133602, 0.04163837,
    0.05096506, 0.05909727, 0.06584432, 0.07104805, 0.0745865, 0.07637669,
    0.07637669, 0.0745865, 0.07104805, 0.06584432, 0.05909727, 0.05096506,
    0.04163837, 0.03133602, 0.02030071, 0.008807], dtype="float32")

    P_grid_lowres = np.array([3.05902319e-07, 8.58439876e-07, 2.40900340e-06,
    6.76027776e-06, 1.89710809e-05, 5.32376871e-05, 1.49398518e-04, 4.19250515e-04,
    1.17652444e-03, 3.30162724e-03, 9.26521700e-03, 2.60005593e-02, 7.29641989e-02,
    2.04756334e-01, 5.74598432e-01, 1.61247122e+00 , 4.52500534e+00,
    1.26983185e+01, 3.56347198e+01, 1.00000290e+02], dtype="float32" )

    T_grid_lowres=np.array([100., 250., 400., 550., 700., 850., 1000.,
    1150., 1300., 1450., 1600., 1750., 1900., 2050., 2200., 2350.,
    2500., 2650., 2800., 2950.], dtype="float32")

    def test_read_kta_lowres(self):
        # test if can read the low res h2o ktable successfully
        gas_id, iso_id, wave_grid, g_ord, del_g, P_grid, T_grid, k_w_g_p_t\
            = read_kta(self.h2o_lowres_table)
        self.assertEqual(gas_id, 1)
        self.assertEqual(iso_id, 0)
        # some tiny disagreements on some arrays (of order 1e-7 relative diff)
        np.testing.assert_array_almost_equal(wave_grid, self.wave_grid_lowres)
        np.testing.assert_array_almost_equal(g_ord, self.g_ord_lowres)
        np.testing.assert_array_almost_equal(del_g, self.del_g_lowres)
        np.testing.assert_array_equal(P_grid, self.P_grid_lowres)
        np.testing.assert_array_equal(T_grid, self.T_grid_lowres)
        # see if k_w_g_p_t has the right shape
        k_w_g_p_t_shape = k_w_g_p_t.shape
        self.assertEqual(k_w_g_p_t_shape[0], len(self.wave_grid_lowres))
        self.assertEqual(k_w_g_p_t_shape[1], len(self.g_ord_lowres))
        self.assertEqual(k_w_g_p_t_shape[2], len(self.P_grid_lowres))
        self.assertEqual(k_w_g_p_t_shape[3], len(self.T_grid_lowres))

    def test_read_kls_lowres(self):
        gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
        k_gas_w_g_p_t = read_kls(self.lowres_files)
        for index, table in enumerate(self.lowres_files):
            self.assertEqual(gas_id_list[index], read_kta(table)[0])
            self.assertEqual(iso_id_list[index], read_kta(table)[1])
        np.testing.assert_array_almost_equal(wave_grid, self.wave_grid_lowres)
        np.testing.assert_array_almost_equal(g_ord, self.g_ord_lowres)
        np.testing.assert_array_almost_equal(del_g, self.del_g_lowres)
        np.testing.assert_array_equal(P_grid, self.P_grid_lowres)
        np.testing.assert_array_equal(T_grid, self.T_grid_lowres)
        # see if k_gas_w_g_p_t shape has the right shape
        k_gas_w_g_p_t_shape = k_gas_w_g_p_t.shape
        self.assertEqual(k_gas_w_g_p_t_shape[0], len(self.lowres_files))
        self.assertEqual(k_gas_w_g_p_t_shape[1], len(self.wave_grid_lowres))
        self.assertEqual(k_gas_w_g_p_t_shape[2], len(self.g_ord_lowres))
        self.assertEqual(k_gas_w_g_p_t_shape[3], len(self.P_grid_lowres))
        self.assertEqual(k_gas_w_g_p_t_shape[4], len(self.T_grid_lowres))

if __name__ == '__main__':
    unittest.main()
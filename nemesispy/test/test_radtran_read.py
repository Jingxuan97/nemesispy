#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np

from nemesispy.radtran.read import read_kta

class TestReadkls(unittest.TestCase):
    h2o = 'test_data/h2o.kta'
    ch4 = 'test_data/ch4.kta'
    wave_grid = np.array(
        [1.14250004, 1.17750001, 1.21249998, 1.24749994, 1.28250003,
        1.3175, 1.35249996, 1.38750005, 1.42250001, 1.45749998, 1.49249995,
        1.52750003, 1.5625, 1.59749997, 1.63250005, 3.5999999,  4.5])
    g_ord = np.array(
        [0.0034357, 0.01801404, 0.04388279, 0.08044151, 0.12683405, 0.18197316,
        0.2445665, 0.31314695, 0.3861071, 0.46173674, 0.53826326, 0.6138929,
        0.68685305, 0.7554335, 0.81802684, 0.87316597, 0.91955847, 0.9561172,
        0.981986, 0.9965643])
    del_g = np.array(
        [0.008807, 0.02030071, 0.03133602, 0.04163837, 0.05096506, 0.05909727,
        0.06584432, 0.07104805, 0.0745865, 0.07637669, 0.07637669, 0.0745865,
        0.07104805, 0.06584432, 0.05909727, 0.05096506, 0.04163837, 0.03133602,
        0.02030071, 0.008807])
    T_grid = np.array(
        [100.0, 250.0, 400.0, 550.0, 700.0, 850.0, 1000.0, 1150.0,
        1300.0, 1450.0, 1600.0, 1750.0, 1900.0, 2050.0, 2200.0,
        2350.0, 2500.0, 2650.0, 2800.0, 2950.0])
    P_grid = np.array(
        [3.0995553e-02,8.6981423e-02,2.4409227e-01,6.8498516e-01,1.9222448e+00,
        5.3943086e+00,1.5137805e+01,4.2480560e+01,1.1921134e+02,3.3453738e+02,
        9.3879810e+02,2.6345066e+03,7.3930977e+03,2.0746936e+04,5.8221188e+04,
        1.6338364e+05,4.5849616e+05,1.2866571e+06,3.6106880e+06,1.0132529e+07])

    # wavelengths =
    def test_invalid_path(self):
        invalid_paths = ['spam','egg','TiO.kta']
        for path in invalid_paths:
            with self.assertRaises(Exception):
                gas_id, iso_id, wave_grid, g_ord, del_g, P_grid, T_grid, \
                    k_w_g_p_t = read_kta(path)

    def test_read_h2o(self):
        gas_id, iso_id, wave_grid, g_ord, del_g, P_grid, T_grid, k_w_g_p_t\
            = read_kta(self.h2o)
        self.assertEqual(gas_id,1)
        self.assertEqual(iso_id,0)
        for iwave in range(len(wave_grid)):
            self.assertAlmostEqual(wave_grid[iwave],self.wave_grid[iwave],
                places=7)
        for ig in range(len(g_ord)):
            self.assertAlmostEqual(g_ord[ig],self.g_ord[ig],
                places=7)
        for ig in range(len(del_g)):
            self.assertAlmostEqual(del_g[ig],self.del_g[ig],
                places=7)
        for ip in range(len(P_grid)):
            self.assertAlmostEqual(0, (P_grid[ip]-self.P_grid[ip])/P_grid[ip],
                places=7)
        for it in range(len(T_grid)):
            self.assertAlmostEqual(0, (T_grid[it]-self.T_grid[it])/T_grid[it],
                places=7)
        self.assertEqual(k_w_g_p_t.shape[0],len(self.wave_grid))
        self.assertEqual(k_w_g_p_t.shape[1],len(self.g_ord))
        self.assertEqual(k_w_g_p_t.shape[2],len(self.P_grid))
        self.assertEqual(k_w_g_p_t.shape[3],len(self.T_grid))

    def test_read_ch4(self):
        gas_id, iso_id, wave_grid, g_ord, del_g, P_grid, T_grid, k_w_g_p_t\
            = read_kta(self.ch4)
        self.assertEqual(gas_id,6)
        self.assertEqual(iso_id,0)
        for iwave in range(len(wave_grid)):
            self.assertAlmostEqual(wave_grid[iwave],self.wave_grid[iwave],
                places=7)
        for ig in range(len(g_ord)):
            self.assertAlmostEqual(g_ord[ig],self.g_ord[ig],
                places=6)
        for ig in range(len(del_g)):
            self.assertAlmostEqual(del_g[ig],self.del_g[ig],
                places=7)
        for ip in range(len(P_grid)):
            self.assertAlmostEqual(0, (P_grid[ip]-self.P_grid[ip])/P_grid[ip],
                places=7)
        for it in range(len(T_grid)):
            self.assertAlmostEqual(0, (T_grid[it]-self.T_grid[it])/T_grid[it],
                places=7)
        self.assertEqual(k_w_g_p_t.shape[0],len(self.wave_grid))
        self.assertEqual(k_w_g_p_t.shape[1],len(self.g_ord))
        self.assertEqual(k_w_g_p_t.shape[2],len(self.P_grid))
        self.assertEqual(k_w_g_p_t.shape[3],len(self.T_grid))

if __name__ == "__main__":
    unittest.main()
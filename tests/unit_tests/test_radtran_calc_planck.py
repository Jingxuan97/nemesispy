#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import unittest
from nemesispy.radtran.calc_planck import calc_planck

class TestRadtranCalcPlanck(unittest.TestCase):

    positive_wave_grid = \
        np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
    zero_wave_grid = \
        np.array([0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
    negative_wave_grid = \
        np.array([ -1,  -2,  -3,  -4,  -5,  -6,  -7,  -8,  -9, -10])
    positive_T = 300
    negative_T = -300

    def test_negative_T(self):
        """Should raise exception when negative temperature is passed"""
        with self.assertRaises(Exception):
            calc_planck(self.positive_wave_grid,self.negative_T)

    def test_zero_wave_grid(self):
        """Should raise exception when wavelengths/numbers contain 0"""
        with self.assertRaises(Exception):
            calc_planck(self.zero_wave_grid,self.positive_T,ispace=0)
        with self.assertRaises(Exception):
            calc_planck(self.zero_wave_grid,self.positive_T,ispace=1)

    def test_negative_wave_grid(self):
        """Should raise exception when negative wavelengths/numbers are passed"""
        with self.assertRaises(Exception):
            calc_planck(self.negative_wave_grid,self.positive_T,ispace=0)
        with self.assertRaises(Exception):
            calc_planck(self.negative_wave_grid,self.positive_T,ispace=1)

if __name__ == '__main__':
    unittest.main()

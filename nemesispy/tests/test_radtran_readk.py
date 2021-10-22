#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from pathlib import Path
from nemesispy.radtran.readk import read_kta, read_kls

nemesispy_path = Path(__file__).parents[1]

lowres_files = ['{}/data/ktables/h2o'.format(nemesispy_path),
         '{}/data/ktables/co2'.format(nemesispy_path),
         '{}/data/ktables/co'.format(nemesispy_path),
         '{}/data/ktables/ch4'.format(nemesispy_path)]
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

    def test_read_kta_lowres_h2o(self):
        # test if can read the low res h2o ktable successfully
        gas_id, iso_id, wave_grid, g_ord, del_g, P_grid, T_grid, k_w_g_p_t\
    = read_kta(lowres_files[0])

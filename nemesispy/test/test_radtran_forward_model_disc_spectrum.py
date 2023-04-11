# -*- coding: utf-8 -*-
import unittest
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


from nemesispy.radtran.forward_model import ForwardModel
from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)
from test_data.planet_wasp_43b import planet


class TestDiscspectrumAccuracy():
    FM = ForwardModel()
    FM.read_input_dict(planet)

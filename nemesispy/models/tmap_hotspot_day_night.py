#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from nemesispy.models.TP_profiles import TP_Guillot

def tmap_hotspot_day_night(P_grid, lon_grid, lat_grid, g_plt, T_eq,
    scale, hot_spot_offset,
    log_kappa_day, log_gamma_day, log_f_day, T_int_day,
    log_kappa_day, log_gamma_day, log_f_day, T_int_day,
    log_kappa_night, log_gamma_night, log_f_night, T_int_night,
    ):
    # scale hard coded to be between 0 and 1
    assert scale <= 1.5 and scale >=0.
    # phase_offset hard coded to be between -90 and 90
    assert hot_spot_offset <=90 and hot_spot_offset >= -90

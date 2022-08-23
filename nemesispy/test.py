#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from nemesispy.common.constants import R_SUN, R_JUP_E, AMU, AU, M_JUP, R_JUP

config = {
    'planet' : {
        'M_plt' : 1.898e27, # kg Jupiter mass
        'R_plt' : 6.9911e7, # m Jupiter radius
        'R_star' : 6.95700e8, # m solar radius
    },
    'gas' : {
        'gas_name_list' : np.array(['H2O','CO2','CO','He','H2']),
        'gas_id_list' : np.array([  1, 2,  5,  6, 40, 39]),
        'iso_id_list' : np.array([0, 0, 0, 0, 0, 0])
    },
    'files' : {
        'opacity' : [
            'h2owasp43.kta',
            'co2wasp43.kta',
            'cowasp43.kta',
            'ch4wasp43.kta'
        ],
        'cia' : 'exocia_hitran12_200-3800K.tab',
    },
    'settings' : {
        'nmu' : 5,
    },
    'atm' : {
        'nlayer' : 20,
        'layer_type' : 'log'
    }
}
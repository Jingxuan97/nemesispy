#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from nemesispy.models.TP_profiles import TP_Guillot
from scipy.special import voigt_profile

def generator(x,Ck0,Ck1,Ck2,Sk1,Sk2,
    Cg0,Cg1,Cg2,Sg1,Sg2,
    Cf,muf,sigmaf,gammaf,
    Ct1,Ct2,mut,gammat):

    y = x/180*np.pi

    kappa = Ck0 + Ck1 * np.cos(y) + Ck2 * np.cos(2*y)\
        + Sk1 * np.sin(y) + Sk2 * np.sin(2*y)

    gamma = Cg0 + Cg1 * np.cos(y) + Cg2 * np.cos(2*y)\
        + Sg1 * np.sin(y) + Sg2 * np.sin(2*y)

    f = Cf * voigt_profile(x-muf,sigmaf,gammaf)

    T_int = Ct1 + Ct2 * ( gammat**2 /( (x-mut)**2 + gammat**2) )

    return kappa, gamma, f, T_int

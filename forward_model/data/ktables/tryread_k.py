# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import FortranFile

name = 'CH4_Katy_ARIEL_test.kta'
f = FortranFile( name, 'r' )
print(f.read_record(dtype='int32' )) 
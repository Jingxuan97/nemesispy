import unittest
import numpy as np
import matplotlib.pyplot as plt
from nemesispy.common.calc_trig import generate_angles, gauss_lobatto_weights

# class TestGenerateAngles(unittest.TestCase):

# class TestGaussLobattoWeights(unittest.TestCase):

nav, wav = gauss_lobatto_weights(phase=90, nmu=4)
lat = wav[0,:]
lon = wav[1,:]
plt.scatter(lon,lat)
plt.savefig('test')
plt.close()
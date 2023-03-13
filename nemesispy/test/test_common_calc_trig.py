import unittest
import numpy as np
import matplotlib.pyplot as plt
from nemesispy.common.calc_trig import arctan, rotatey, rotatez, \
    generate_angles, gauss_lobatto_weights

class TestTrigonometry(unittest.TestCase):

    def test_arctan(self):
        # test the arctan routine
        origin = [0,0]
        north = [0,1]
        south = [0,-1]
        east = [1,0]
        west = [-1,0]

        arctan_origin = arctan(origin[0],origin[1])
        arctan_north = arctan(north[0],north[1])
        arctan_south = arctan(south[0],south[1])
        arctan_east = arctan(east[0],east[1])
        arctan_west = arctan(west[0],west[1])

        self.assertEqual(arctan_origin,0.0)
        self.assertEqual(arctan_north,0.5*np.pi)
        self.assertEqual(arctan_south,1.5*np.pi)
        self.assertEqual(arctan_east,0.0)
        self.assertEqual(arctan_west,1.0*np.pi)

# class TestGenerateAngles(unittest.TestCase):

# class TestGaussLobattoWeights(unittest.TestCase):

# nav, wav = gauss_lobatto_weights(phase=90, nmu=4)
# lat = wav[0,:]
# lon = wav[1,:]
# plt.scatter(lon,lat)
# plt.savefig('test')
# plt.close()

if __name__ == "__main__":
    unittest.main()
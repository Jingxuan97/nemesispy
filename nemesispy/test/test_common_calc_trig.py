import unittest
import numpy as np
import matplotlib.pyplot as plt
from nemesispy.common.calc_trig import arctan, rotatey, rotatez, \
    generate_angles, gauss_lobatto_weights
from nemesispy.common.calc_trig_new import disc_weights, \
    add_azimuthal_weights_2tp

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

# nav, wav = disc_weights(phase=180, nmu=6)
# lat = wav[0,:]
# lon = wav[1,:]
# plt.scatter(lon,lat)
# plt.savefig('test_disc')
# plt.close()


nmu = 6
daybound1 = -90
daybound2 = 90
for phase in [0,30,60,90,120,150,180,210,240,270,300,330,360]:

    day_lat, day_lon, night_lat, night_lon, \
        new_day_lat, new_day_lon, new_night_lat, new_night_lon \
        = add_azimuthal_weights_2tp(phase, nmu, daybound1, daybound2)
    plt.xlim(0,360)
    plt.ylim(0,90)
    plt.scatter(day_lon,day_lat,color='red')
    plt.scatter(night_lon,night_lat,color='blue')
    plt.scatter(new_night_lon, new_night_lat,color='green')
    plt.scatter(new_day_lon, new_day_lat,color='yellow')
    plt.axvline(90)
    plt.axvline(270)
    plt.savefig('split{}'.format(phase))
    plt.close()
if __name__ == "__main__":
    unittest.main()
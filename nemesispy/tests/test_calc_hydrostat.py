import numpy as np
import scipy.interpolate
from nemesispy.common.calc_hydrostat import calc_hydrostat, calc_grav_simple
from nemesispy.common.constants import R_JUP, M_JUP, AMU, BAR, G, K_B

import unittest

def isothermal_height(P,T,mmw,M_plt,R_plt):
    P0 = P[0]
    height = np.zeros(P.shape)
    for i in np.arange(1,len(P)):
        g = calc_grav_simple(height[i-1],M_plt,R_plt)
        z = np.log(P0/P[i]) * K_B * T  / (mmw * g)
        height[i] = z
    return height

class TestCalcHydrostat(unittest.TestCase):
    NLAYER = 20
    pressure = np.geomspace(50*BAR,1e-3*BAR,NLAYER)
    mmw = np.ones(NLAYER) * 2 * AMU
    R_plt = 1 * R_JUP
    M_plt = 1 * M_JUP

    def test_isothermal(self):
        result = np.array(
            [      0.        ,   91458.94741604,  183157.50477109,
        275096.61386969,  367277.22075478,  459700.27572779,
        552366.73336902,  645277.55255797,  738433.69649384,
        831836.13271612,  925485.83312529, 1019383.77400358,
       1113530.93603586, 1207928.30433068, 1302576.86844132,
       1397477.622387  , 1492631.56467422, 1588039.69831812,
       1683703.03086401, 1779622.57440902])
        T = 1e3
        iso_T = np.ones(self.NLAYER)*T
        height = calc_hydrostat(
            self.pressure,iso_T,self.mmw,self.M_plt,self.R_plt
        )
        for ilayer in range(self.NLAYER):
            self.assertAlmostEqual(
                height[ilayer],result[ilayer],places=2
            )


class TestCalcGravSimple(unittest.TestCase):
    # Test Data
    height = 0
    mercury_mass = 3.30e23
    mercury_radius = 2.44e6
    mercury_gravity = 3.7
    earth_mass = 5.97e24
    earth_radius = 6.38e6
    earth_gravity = 9.79
    jupiter_mass = 1.90e27
    jupiter_radius = 7.15e7
    jupiter_gravity = 24.8

    # 100km to 1000km
    height_array = np.array(
        [      0.,  100000.,  200000.,  300000.,  400000.,  500000.,
        600000.,  700000.,  800000.,  900000., 1000000.])
    mercury_g_array = np.array(
        [3.69947427, 3.41391128, 3.16017992, 2.93371916, 2.73075655,
       2.54815008, 2.38326588, 2.23388271, 2.09811671, 1.97436176,
       1.86124172])
    earth_g_array = np.array(
        [9.78900831, 9.48920968, 9.20297554, 8.92949976, 8.66803522,
       8.41788868, 8.17841623, 7.94901908, 7.72913987, 7.51825923,
       7.31589277])
    jupiter_g_array = np.array(
        [24.80545748, 24.73621688, 24.66726578, 24.59860259, 24.53022568,
       24.46213349, 24.39432442, 24.32679691, 24.2595494 , 24.19258035,
       24.12588823])

    def test_real_number_input(self):
        """Calculate gravity for real number inputs"""
        mercury_g = calc_grav_simple(
            self.height,self.mercury_mass,self.mercury_radius)
        self.assertAlmostEqual(
            self.mercury_gravity,mercury_g,places=2)
        earth_g = calc_grav_simple(
            self.height,self.earth_mass,self.earth_radius)
        self.assertAlmostEqual(
            self.earth_gravity,earth_g,places=2)
        jupiter_g = calc_grav_simple(
            self.height,self.jupiter_mass,self.jupiter_radius)
        self.assertAlmostEqual(
            self.jupiter_gravity,jupiter_g,places=1)

    def test_real_array_input(self):
        """Calculate gravity for real array inputs"""
        mercury_g = calc_grav_simple(
            self.height_array,self.mercury_mass,self.mercury_radius)
        earth_g = calc_grav_simple(
            self.height_array,self.earth_mass,self.earth_radius)
        jupiter_g = calc_grav_simple(
            self.height_array,self.jupiter_mass,self.jupiter_radius)
        for i in range(len(self.height_array)):
            self.assertAlmostEqual(
                self.mercury_g_array[i],mercury_g[i],places=2)
            self.assertAlmostEqual(
                self.earth_g_array[i],earth_g[i],places=2)
            self.assertAlmostEqual(
                self.jupiter_g_array[i],jupiter_g[i],places=2)

if __name__ == "__main__":
    unittest.main()
import unittest
import numpy as np

from nemesispy.radtran.calc_planck import calc_planck

class TestCalcPlanck(unittest.TestCase):
    wavelengths = np.linspace(1,10,num=50)
    T = 1000

    def test_zero_temperature(self):
        T_zero = 0
        bb = calc_planck(self.wavelengths,T_zero)
        for iwave in range(len(bb)):
            self.assertEqual(bb[iwave],0)

    def test_invalid_temperature(self):
        invalid_T = -1
        with self.assertRaises(Exception):
            bb = calc_planck(self.wavelengths,-1)

    def test_invalid_spectral_type(self):
        invalid = [-1,2,3]
        for type in invalid:
            with self.assertRaises(Exception):
                bb = calc_planck(self.wavelengths,self.T,
                    ispace=type)

    def test_invalid_wavelenghts(self):
        invalid_wavelength = [[0],[0,-1],[1,3,0]]
        for wavelength in invalid_wavelength:
            with self.assertRaises(Exception):
                bb = calc_planck(wavelength,self.T)

if __name__ == "__main__":
    unittest.main()
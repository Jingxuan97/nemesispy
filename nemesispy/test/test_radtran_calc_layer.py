import unittest
import numpy as np

from nemesispy.radtran.calc_layer import split

class TestSplit(unittest.TestCase):
    NLAYER = 20
    H_model = np.array(
        [      0.        ,   52631.57894737,  105263.15789474,
        157894.73684211,  210526.31578947,  263157.89473684,
        315789.47368421,  368421.05263158,  421052.63157895,
        473684.21052632,  526315.78947368,  578947.36842105,
        631578.94736842,  684210.52631579,  736842.10526316,
        789473.68421053,  842105.26315789,  894736.84210526,
        947368.42105263, 1000000.        ]) # m
    P_model = np.array(
        [1.00000000e+07, 5.45559478e+06, 2.97635144e+06, 1.62377674e+06,
        8.85866790e+05, 4.83293024e+05, 2.63665090e+05, 1.43844989e+05,
        7.84759970e+04, 4.28133240e+04, 2.33572147e+04, 1.27427499e+04,
        6.95192796e+03, 3.79269019e+03, 2.06913808e+03, 1.12883789e+03,
        6.15848211e+02, 3.35981829e+02, 1.83298071e+02, 1.00000000e+02]) # pa
        # P_model = np.geomspace(1e7,1e2,20)

    def test_H0_out_of_bounds(self):
        out_of_bounds_H0 = [self.H_model[0]-1, self.H_model[-1]+1]
        for H0 in out_of_bounds_H0:
            with self.assertRaises(AssertionError):
                split(self.H_model,self.P_model,self.NLAYER,
                    H_0 = H0)

    def test_decreasing_H_model(self):
        decreasing_H = self.H_model[::-1]
        with self.assertRaises(AssertionError):
            split(decreasing_H,self.P_model,self.NLAYER)

    def test_invaid_layer_scheme(self):
        invalid_type_eg = [-1,6,7,8,9,10]
        for invalid_type in invalid_type_eg:
            with self.assertRaises(Exception):
                split(self.H_model,self.P_model,self.NLAYER,
                    layer_type=invalid_type)

    def test_too_few_layers(self):
        invalid_NLAYER = [0,1]
        for NLAYER in invalid_NLAYER:
            with self.assertRaises(AssertionError):
                split(self.H_model,self.P_model,NLAYER)

    def test_layer_type_1(self):
        NLAYER = 10
        P_base_truth = np.geomspace(1e7,1e2,NLAYER+1)[:-1]
        H_base_truth = np.interp(P_base_truth,
            self.P_model[::-1], self.H_model[::-1])
        H_base,P_base =  split(self.H_model, self.P_model, NLAYER,
            layer_type=1)
        for ilayer in range(NLAYER):
            self.assertAlmostEqual(P_base_truth[ilayer], P_base[ilayer],
                places=4)
            self.assertAlmostEqual(H_base_truth[ilayer], H_base[ilayer],
                places=4)

if __name__ == "__main__":
    unittest.main()
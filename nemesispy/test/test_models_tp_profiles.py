import unittest
import numpy as np
from nemesispy.models.TP_profiles import TP_Guillot, TP_Guillot14

class TestTPGuillot(unittest.TestCase):
    P_highest = 20 * 1e5
    P_lowest = 1e-3 * 1e5
    NLAYER = 20
    P_range = np.geomspace(P_highest,P_lowest,NLAYER)
    g_plt = 25
    T_eq = 1000
    eg_params = {
        'k_IR' : 1e-3,
        'gamma' : 1e-2,
        'f' : 0.5,
        'T_int' : 200,
    }
    hot_params = {
        'k_IR' : 5e-1,
        'gamma' : 5e-1,
        'f' : 100,
        'T_int' : 10000,
    }
    def test_invalid_inputs(self):
        # catch negative inputs
        invalid_k_ir = -1
        with self.assertRaises(AssertionError):
            TP = TP_Guillot(
                self.P_range,
                self.g_plt,
                self.T_eq,
                k_IR = invalid_k_ir,
                gamma = self.eg_params['gamma'],
                f = self.eg_params['f'],
                T_int = self.eg_params['T_int']
                )
        invalid_gamma = -1
        with self.assertRaises(AssertionError):
            TP = TP_Guillot(
                self.P_range,
                self.g_plt,
                self.T_eq,
                k_IR = self.eg_params['k_IR'],
                gamma = invalid_gamma,
                f = self.eg_params['f'],
                T_int = self.eg_params['T_int']
                )
        invalid_f = -1
        with self.assertRaises(AssertionError):
            TP = TP_Guillot(
                self.P_range,
                self.g_plt,
                self.T_eq,
                k_IR = self.eg_params['k_IR'],
                gamma = self.eg_params['gamma'],
                f = invalid_f,
                T_int = self.eg_params['T_int']
                )
        invalid_T_int = -1
        with self.assertRaises(AssertionError):
            TP = TP_Guillot(
                self.P_range,
                self.g_plt,
                self.T_eq,
                k_IR = self.eg_params['k_IR'],
                gamma = self.eg_params['gamma'],
                f = self.eg_params['f'],
                T_int = invalid_T_int
                )

    def test_temperature_upper_bound(self):
        # temperature upper limit is 9999 K
        TP = TP_Guillot(
            self.P_range,
            self.g_plt,
            self.T_eq,
            k_IR = self.hot_params['k_IR'],
            gamma = self.hot_params['gamma'],
            f = self.hot_params['f'],
            T_int = self.hot_params['T_int']
            )
        for ilayer, T in enumerate(TP):
            self.assertEqual(T,9999)

    def test_calculation(self):
        TP = TP_Guillot(
            self.P_range,
            self.g_plt,
            self.T_eq,
            k_IR = self.eg_params['k_IR'],
            gamma = self.eg_params['gamma'],
            f = self.eg_params['f'],
            T_int = self.eg_params['T_int']
            )
        truths = np.array(
            [2850.6661796929848, 2654.1788753056735, 2423.7518677330154,
             2185.990060249486, 1958.859984562856, 1752.6417999462221,
             1572.6158173981728, 1421.1578941279556, 1298.7946278681284,
             1204.4346617619512, 1135.30544407448, 1087.1960816069436,
             1055.2187455459755, 1034.7326521274335, 1021.9567830721173,
             1014.1338854684531, 1009.4002670318981, 1006.5571771930058,
             1004.8573414325596, 1003.8438420242613]
        )
        for ilayer in range(len(TP)):
            self.assertAlmostEqual(TP[ilayer],truths[ilayer])


class TestTPParmentier(unittest.TestCase):
    P_highest = 20 * 1e5
    P_lowest = 1e-3 * 1e5
    NLAYER = 20
    P_range = np.geomspace(P_highest,P_lowest,NLAYER)
    g_plt = 25
    T_eq = 1000
    eg_params = {
        'k_IR' : 1e-3,
        'gamma1' : 1e-2,
        'gamma2' : 1e-1,
        'alpha' : 0.5,
        'beta' : 0.5,
        'T_int' : 200,
    }
    hot_params = {
        'k_IR' : 1e-2,
        'gamma1' : 1e-1,
        'gamma2' : 1e-1,
        'alpha' : 0.5,
        'beta' : 100,
        'T_int' : 5000,
    }
    def test_invalid_inputs(self):
        # catch negative inputs
        invalid_k_ir = -1
        with self.assertRaises(AssertionError):
            TP = TP_Guillot14(
                self.P_range,
                self.g_plt,
                self.T_eq,
                k_IR = invalid_k_ir,
                gamma1 = self.eg_params['gamma1'],
                gamma2 = self.eg_params['gamma2'],
                alpha = self.eg_params['alpha'],
                beta = self.eg_params['beta'],
                T_int = self.eg_params['T_int']
            )
        invalid_gamma1 = [0,-1]
        for gamma1 in invalid_gamma1:
            with self.assertRaises(AssertionError):
                TP = TP_Guillot14(
                    self.P_range,
                    self.g_plt,
                    self.T_eq,
                    k_IR = self.eg_params['k_IR'],
                    gamma1 = gamma1,
                    gamma2 = self.eg_params['gamma2'],
                    alpha = self.eg_params['alpha'],
                    beta = self.eg_params['beta'],
                    T_int = self.eg_params['T_int']
                )
        invalid_gamma2 = [0,-1]
        for gamma2 in invalid_gamma2:
            with self.assertRaises(AssertionError):
                TP = TP_Guillot14(
                    self.P_range,
                    self.g_plt,
                    self.T_eq,
                    k_IR = self.eg_params['k_IR'],
                    gamma1 = self.eg_params['gamma1'],
                    gamma2 = gamma2,
                    alpha = self.eg_params['alpha'],
                    beta = self.eg_params['beta'],
                    T_int = self.eg_params['T_int']
                )
        invalid_alpha = [-0.1,1.1]
        for alpha in invalid_alpha:
            with self.assertRaises(AssertionError):
                TP = TP_Guillot14(
                    self.P_range,
                    self.g_plt,
                    self.T_eq,
                    k_IR = self.eg_params['k_IR'],
                    gamma1 = self.eg_params['gamma1'],
                    gamma2 = self.eg_params['gamma2'],
                    alpha = alpha,
                    beta = self.eg_params['beta'],
                    T_int = self.eg_params['T_int']
                )
        invalid_beta = -1
        with self.assertRaises(AssertionError):
            TP = TP_Guillot14(
                self.P_range,
                self.g_plt,
                self.T_eq,
                k_IR = self.eg_params['k_IR'],
                gamma1 = self.eg_params['gamma1'],
                gamma2 = self.eg_params['gamma2'],
                alpha = self.eg_params['alpha'],
                beta = invalid_beta,
                T_int = self.eg_params['T_int']
            )
        invalid_T_int = -1
        with self.assertRaises(AssertionError):
            TP = TP_Guillot14(
                self.P_range,
                self.g_plt,
                self.T_eq,
                k_IR = self.eg_params['k_IR'],
                gamma1 = self.eg_params['gamma1'],
                gamma2 = self.eg_params['gamma2'],
                alpha = self.eg_params['alpha'],
                beta = self.eg_params['beta'],
                T_int = invalid_T_int
            )

    def test_temperature_upper_bound(self):
        # temperature upper limit is 9999 K
        TP = TP_Guillot14(
            self.P_range,
            self.g_plt,
            self.T_eq,
            k_IR = self.hot_params['k_IR'],
            gamma1 = self.hot_params['gamma1'],
            gamma2 = self.hot_params['gamma2'],
            alpha = self.hot_params['alpha'],
            beta = self.hot_params['beta'],
            T_int = self.hot_params['T_int']
            )
        for ilayer, T in enumerate(TP):
            self.assertEqual(T,9999)

    def test_calculation(self):
        TP = TP_Guillot14(
            self.P_range,
            self.g_plt,
            self.T_eq,
            k_IR = self.eg_params['k_IR'],
            gamma1 = self.eg_params['gamma1'],
            gamma2 = self.eg_params['gamma2'],
            alpha = self.eg_params['alpha'],
            beta = self.eg_params['beta'],
            T_int = self.eg_params['T_int']
            )
        truths = np.array(
        [1147.623735468425, 1072.3934485423933, 992.9888680210805,
         914.290873422902, 837.7609798471251, 764.5184043258015,
         696.6726845861475, 636.6742344165456, 586.4186311765807,
         546.736099501995, 517.2479167202697, 496.5662842618649,
         482.76893438634625, 473.9185271732182, 468.39961957736904,
         465.0231543589519, 462.9826332576048, 461.75888094144847,
         461.02839692552425, 460.5935887848739]
        )
        for ilayer in range(len(TP)):
            self.assertAlmostEqual(TP[ilayer],truths[ilayer])

if __name__ == "__main__":
    unittest.main()
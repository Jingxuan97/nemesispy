#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from nemesispy.disc.trig import arctan, rotatey, rotatez, generate_angles, \
    gauss_lobatto_weights

class TestDisc(unittest.TestCase):

    def test_arctan(self):
        # test if arctan return the argument of a point in the range of [0,2pi)
        self.assertEqual(arctan(0,0),0*np.pi)
        self.assertEqual(arctan(1,0),0*np.pi)
        self.assertEqual(arctan(1,1),0.25*np.pi)
        self.assertEqual(arctan(0,1),0.5*np.pi)
        self.assertEqual(arctan(-1,1),0.75*np.pi)
        self.assertEqual(arctan(-1,0),1*np.pi)
        self.assertEqual(arctan(-1,-1),1.25*np.pi)
        self.assertEqual(arctan(0,-1),1.5*np.pi)
        self.assertEqual(arctan(1,-1),1.75*np.pi)

    def test_rotatey(self):
        # test if rotatey correctly rotate a vector around the y-axis
        for vector in [[1,0,0],[0,1,0],[0,0,1]]:
            np.testing.assert_array_equal(rotatey(vector,0),vector)
            np.testing.assert_array_equal(np.around(rotatey(vector,2*np.pi),10),vector)

        np.testing.assert_array_equal(np.around(rotatey([1,0,0],0.5*np.pi),10),[0,0,-1])
        np.testing.assert_array_equal(np.around(rotatey([1,0,0],1*np.pi),10),[-1,0,0])
        np.testing.assert_array_equal(np.around(rotatey([1,0,0],1.5*np.pi),10),[0,0,1])

        np.testing.assert_array_equal(np.around(rotatey([0,1,0],0.5*np.pi),10),[0,1,0])
        np.testing.assert_array_equal(np.around(rotatey([0,1,0],1*np.pi),10),[0,1,0])
        np.testing.assert_array_equal(np.around(rotatey([0,1,0],1.5*np.pi),10),[0,1,0])

        np.testing.assert_array_equal(np.around(rotatey([0,0,1],0.5*np.pi),10),[1,0,0])
        np.testing.assert_array_equal(np.around(rotatey([0,0,1],1*np.pi),10),[0,0,-1])
        np.testing.assert_array_equal(np.around(rotatey([0,0,1],1.5*np.pi),10),[-1,0,0])

    def test_rotatez(self):
        # test if rotatez correctly rotate a vector around the z-axis
        for vector in [[1,0,0],[0,1,0],[0,0,1]]:
            np.testing.assert_array_equal(rotatez(vector,0),vector)
            np.testing.assert_array_equal(np.around(rotatez(vector,2*np.pi),10),vector)

        np.testing.assert_array_equal(np.around(rotatez([1,0,0],0.5*np.pi),10),[0,1,0])
        np.testing.assert_array_equal(np.around(rotatez([1,0,0],1*np.pi),10),[-1,0,0])
        np.testing.assert_array_equal(np.around(rotatez([1,0,0],1.5*np.pi),10),[0,-1,0])

        np.testing.assert_array_equal(np.around(rotatez([0,1,0],0.5*np.pi),10),[-1,0,0])
        np.testing.assert_array_equal(np.around(rotatez([0,1,0],1*np.pi),10),[0,-1,0])
        np.testing.assert_array_equal(np.around(rotatez([0,1,0],1.5*np.pi),10),[1,0,0])

        np.testing.assert_array_equal(np.around(rotatez([0,0,1],0.5*np.pi),10),[0,0,1])
        np.testing.assert_array_equal(np.around(rotatez([0,0,1],1*np.pi),10),[0,0,1])
        np.testing.assert_array_equal(np.around(rotatez([0,0,1],1.5*np.pi),10),[0,0,1])

    def test_generate_angles(self):
        # more work need to be done here to test properly
        zen, azi, lat, lon = generate_angles(180,0,0)

    def test_gauss_lobatto_weights(self):
        # more work need to be done here to test properly

        # test invalid quadrature ring number input (invalid nmu)
        with self.assertRaises(Exception) as cm:
            gauss_lobatto_weights(phase=0, nmu=0)
        self.assertEqual("Need at least 2 quadrature rings", str(cm.exception))
        with self.assertRaises(Exception) as cm:
            gauss_lobatto_weights(phase=0, nmu=1)
        self.assertEqual("Need at least 2 quadrature rings", str(cm.exception))
        with self.assertRaises(Exception) as cm:
            gauss_lobatto_weights(phase=0, nmu=6)
        self.assertEqual("Currently cannot do more than 5 quadrature rings", str(cm.exception))

"""
for i in range(10000):
    test = np.random.rand(1,3)*np.array([360, 1, 360])
    test = test[0]
    phase = test[0]
    rho = test[1]
    alpha = test[2]
    v1 = generate_angles(phase, rho, alpha)
    v2 = thetasol_gen_azi_latlong(phase, rho, alpha)
    diff = (np.array(v2) - np.array(v1))
    if sum(diff) != 0:
        raise
"""

"""
for phase in np.linspace(0,360):
    dtr = np.pi/180
    z_term = np.linspace(-1,1,201)
    theta_term = 2*np.pi - np.arccos(z_term)
    if 0<= phase <= 180:
        theta_term = 2*np.pi - np.arccos(z_term)
    else:
        theta_term = np.arccos(z_term)
    x_term = np.around(np.sin(theta_term),14) * np.around(np.cos(phase*dtr),14)

    alpha =  np.arccos(z_term)+np.pi/2.
    minus_r_sin_theta = np.around(np.cos(alpha),14)
    old_x_term = minus_r_sin_theta * np.around(np.cos((phase)*np.pi/180.), 14)
    if (phase > 180.0):
        old_x_term = -old_x_term

    # print(old_x_term-x_term)
    print(sum(old_x_term-x_term))
"""

"""
for phase in np.linspace(0,360):
    diff = gauss_lobatto_weights(phase,3)[1] - subdiscweightsv3(phase,3)[1]
    print(diff)

for phase in np.linspace(0,360):
    diff = gauss_lobatto_weights(phase,5)[1] - subdiscweightsv3(phase,5)[1]
    print(diff)
"""

if __name__ == '__main__':
    unittest.main()
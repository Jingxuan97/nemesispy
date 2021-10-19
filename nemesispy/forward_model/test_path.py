#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from path import split, average

class TestSplit(unittest.TestCase):
    
    def test_undefined_scheme(self):
        # should stop code when layering scheme is not defined
        Nlayer = 100
        Nprofile = 100
        H = np.linspace(0,1e6,Nprofile)
        P = np.linspace(1e5,1,Nprofile)
        layer_type = 6
        with self.assertRaises(Exception):
            H_base, P_base = split(H,P,Nlayer,layer_type=layer_type)

    def test_layer_bottom_height_range(self):
        # should stop code bottom_height is out of range
        Nlayer = 100
        Nprofile = 100
        H = np.linspace(0,1e6,Nprofile)
        P = np.linspace(1e5,1,Nprofile)
        with self.assertRaises(AssertionError):
            H_base, P_base = split(H,P,Nlayer,bottom_height=(H[0]-1))
        with self.assertRaises(AssertionError):
            H_base, P_base = split(H,P,Nlayer,bottom_height=(H[-1]))

    def test_path_angle_range(self):
        # test spliting by equal line-of-sight path intervals
        Nlayer = 100
        Nprofile = 100
        H = np.linspace(0,1e6,Nprofile)
        P = np.linspace(1e5,1,Nprofile)
        layer_type = 3
        # code should stop if zenith angle is not between [0,90]
        with self.assertRaises(AssertionError):
            H_base, P_base = split(H,P,Nlayer,layer_type,path_angle=-1)
        with self.assertRaises(AssertionError):
            H_base, P_base = split(H,P,Nlayer,layer_type,path_angle=91)   
    
    def test_layer_type_0(self):
        # test spliting a profile by equal changes in pressure
        Nlayer = 50
        Nprofile = 100
        H = np.linspace(0,1e6,Nprofile)
        P = np.linspace(1e5,1,Nprofile)
        layer_type = 0
        H_base, P_base = split(H,P,Nlayer,layer_type)
        del_P = P_base[:-1]-P_base[1:]
        del_P = np.round(del_P,5)
        result = np.all(del_P == del_P[0])
        self.assertTrue(result)

    def test_layer_type_1(self):
        # test spliting a profile by equal changes in log pressure
        Nlayer = 100
        Nprofile = 100
        H = np.linspace(0,1e6,Nprofile)
        P = np.linspace(1e5,1,Nprofile)
        layer_type = 1
        H_base, P_base = split(H,P,Nlayer,layer_type)
        del_log_P = np.log(P_base[:-1])-np.log(P_base[1:])
        del_log_P = np.round(del_log_P,5)
        result = np.all(del_log_P == del_log_P[0])
        self.assertTrue(result)

    def test_layer_type_2(self):
        # test spliting by equal height intervals
        Nlayer = 100
        Nprofile = 100
        H = np.linspace(0,1e6,Nprofile)
        P = np.linspace(1e5,1,Nprofile)
        layer_type = 2
        H_base, P_base = split(H,P,Nlayer,layer_type)
        del_H = H_base[:-1]-H_base[1:]
        del_H = np.round(del_H,5)
        result = np.all(del_H == del_H[0])
        self.assertTrue(result)
    
    def test_layer_type_3(self):
        # test spliting by equal line-of-sight path intervals
        Nlayer = 100
        Nprofile = 100
        H = np.linspace(0,1e6,Nprofile)
        P = np.linspace(1e5,1,Nprofile)
        layer_type = 3
        angles = np.linspace(0,90,100)
        radius = 7e7
        for i in angles:
            H_base, P_base \
                = split(H,P,Nlayer,layer_type,path_angle=i,radius=radius)
            z0 = radius 
            sin = np.sin(i*np.pi/180)
            cos = np.cos(i*np.pi/180)
            S_base = np.sqrt((H_base+z0)**2-(z0*sin)**2)-z0*cos
            del_S = S_base[1:]-S_base[:-1]
            del_S = np.round(del_S,4)
            result = np.all(del_S == del_S[0])
            self.assertTrue(result) 

    def test_layer_type_4(self):
        # layer base pressure levels specified by P_base
        Nlayer = 100
        Nprofile = 100
        H = np.linspace(0,1e6,Nprofile)
        P = np.linspace(1e5,1,Nprofile)
        layer_type = 4
        # code should stop if no input base pressures are given
        with self.assertRaises(AssertionError):
            H_base,P_base = split(H,P,Nlayer,layer_type)
        # code should stop if input base pressures out of range of profile
        with self.assertRaises(AssertionError):
            P_base = np.geomspace(P[0]+1,P[-1]*10,Nlayer)
            H_base,P_base = split(H,P,Nlayer,layer_type,P_base=P_base)
        with self.assertRaises(AssertionError):
            P_base = np.geomspace(P[0],P[-1]*0.1,Nlayer)
            H_base,P_base = split(H,P,Nlayer,layer_type,P_base=P_base)
        P_base = np.geomspace(P[0]-1,P[-1]+1,Nlayer)
        H_base,P_base = split(H,P,Nlayer,layer_type,P_base=P_base)

    def test_layer_type_5(self):
        # layer base pressure levels specified by P_base
        Nlayer = 100
        Nprofile = 100
        H = np.linspace(0,1e6,Nprofile)
        P = np.linspace(1e5,1,Nprofile)
        layer_type = 5
        # code should stop if no input base pressures are given
        with self.assertRaises(AssertionError):
            H_base,P_base = split(H,P,Nlayer,layer_type)
        # code should stop if input base heights out of range of profile
        with self.assertRaises(AssertionError):
            H_base = np.linspace(H[0]-1,H[-1],Nlayer)
            H_base,P_base = split(H,P,Nlayer,layer_type,H_base=H_base)
        with self.assertRaises(AssertionError):
            H_base = np.linspace(H[0],H[-1]+1,Nlayer)
            H_base,P_base = split(H,P,Nlayer,layer_type,H_base=H_base)
        H_base = np.linspace(H[0]+1,H[-1]-1,Nlayer)
        H_base,P_base = split(H,P,Nlayer,layer_type,H_base=H_base)


class TestAverage(unittest.TestCase):
    
    def test_integration_type0_nadir(self):
        radius = 7e7
        Natm = 100
        H_atm = np.linspace(0,1e6,Natm)
        P_atm = np.linspace(1e6,1e1,Natm)
        T_atm = np.linspace(4e4,1e4,Natm)**0.75
        VMR1 = np.ones(Natm)*1e-1
        VMR2 = np.ones(Natm)*2e-4
        VMR3 = np.ones(Natm)*3e-4
        VMR4 = np.zeros(Natm)*4e-4
        VMR5 = (1-VMR1-VMR2-VMR3-VMR4)*0.15
        VMR6 = VMR5*0.85/0.15
        VMR_atm = np.vstack((VMR1,VMR2,VMR3,VMR4,VMR5,VMR6)).T
        ID = [1,2,5,6,40,39]
        Nlayer = 50
        H_base = np.geomspace(10,9.9e5,Nlayer)
        path_angle = 0
        integration_type = 0
        H_layer,P_layer,T_layer,VMR_layer,total_path,path,scale\
            = average(radius, H_atm, P_atm, T_atm, VMR_atm, ID, H_base, path_angle,\
                      integration_type=integration_type)
        # layer scaling factor should be 1 if path angle = 0
        for i in range(Nlayer):
            self.assertEqual(np.round(scale[i],4),1)
    
    def test_integration_type1_nadir(self):
        radius = 7e7
        Natm = 100
        H_atm = np.linspace(0,1e6,Natm)
        P_atm = np.linspace(1e6,1e1,Natm)
        T_atm = np.linspace(4e4,1e4,Natm)**0.75
        VMR1 = np.ones(Natm)*1e-1
        VMR2 = np.ones(Natm)*2e-4
        VMR3 = np.ones(Natm)*3e-4
        VMR4 = np.zeros(Natm)*4e-4
        VMR5 = (1-VMR1-VMR2-VMR3-VMR4)*0.15
        VMR6 = VMR5*0.85/0.15
        VMR_atm = np.vstack((VMR1,VMR2,VMR3,VMR4,VMR5,VMR6)).T
        ID = [1,2,5,6,40,39]
        Nlayer = 50
        H_base = np.geomspace(10,9.9e5,Nlayer)
        path_angle = 0
        integration_type = 1
        H_layer,P_layer,T_layer,VMR_layer,total_path,path,scale\
            = average(radius, H_atm, P_atm, T_atm, VMR_atm, ID, H_base, path_angle,\
                      integration_type=integration_type)
        # layer scaling factor should be 1 if path angle = 0
        for i in range(Nlayer):
            self.assertEqual(np.round(scale[i],4),1)


if __name__ == '__main__':
    unittest.main()

"""
def test_split():
    import matplotlib.pyplot as plt
    H_atm = np. array([0.,  373687.,  725112., 1035879., 1308209., 1553557.,
        1785055., 2013748., 2237765., 2471070.])
    P_atm = np.array([1.00000e+06, 3.59381e+05, 1.29155e+05, 4.64160e+04, 1.66810e+04,
        5.99500e+03, 2.15400e+03, 7.74000e+02, 2.78000e+02, 1.00000e+02])
    Nlayer = 10
    layer_type = [0,1,2]
    interp_type = 1
    for i in layer_type:
        H_base, P_base = split(H_atm,P_atm,Nlayer=Nlayer,layer_type=i,interp_type=interp_type)
        plt.scatter(H_base*1e-3,P_base*1e-5,s=10,marker='x',color='k')
        plt.gca().invert_yaxis()
        plt.yscale('log')
        plt.ylabel('Pressure(bar)')
        plt.xlabel('Height(km)')
        plt.title('layer_type {}'.format(i))
        plt.plot(H_atm*1e-3,P_atm*1e-5)
        plt.show()
test_split()
"""
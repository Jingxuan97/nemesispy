import unittest
import numpy as np

from nemesispy.common.interpolate_gcm import interp_gcm_X,interp_gcm
from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)



# for ilon, lon in enumerate(xlon):
#     for ilat, lat in enumerate(xlat):
#         a  = interp_gcm_X(lon,lat,pv*1.1,
#                 xlon,xlat,pv,tmap,0)
#         b = tmap[ilon,ilat]
#         c = (a-b)/b

a  = interp_gcm(0,90,pv,xlon,xlat,pv,tmap,vmrmap,0)

#         print('interpolated',a)
#         print('real profile',b)
#         print('difference',c)

# N_pole =  interp_gcm_X(0,91,pv*1.1,
#             xlon,xlat,pv,tmap,0)
# print('N_pole',N_pole)

# S_pole =  interp_gcm_X(0,-90,pv,
#             xlon,xlat,pv,tmap,0)
# print('S_pole',S_pole)

# N_pole =  interp_gcm_X(0,90,pv,
#             xlon,xlat,pv,tmap,0)
# print('N_pole',N_pole)

# anti =  interp_gcm_X(-180,0,pv*1.1,
#             xlon,xlat,pv,tmap,0)
# print('anti',anti)


class TestInterpolateGCMX(unittest.TestCase):

    gcm_lon = xlon
    gcm_lat = xlat
    gcm_p = pv
    X = tmap

    def test_interpolate_out_of_bounds(self):
        with self.assertRaises(AssertionError):
            over_north_pole =  interp_gcm_X(0,91,self.gcm_p,self.gcm_lon,
                self.gcm_lat,self.gcm_p,self.X,0)
        with self.assertRaises(AssertionError):
            under_south_pole =  interp_gcm_X(0,-91,self.gcm_p,self.gcm_lon,
                self.gcm_lat,self.gcm_p,self.X,0)

    def test_interpolate_at_grid_points(self):
        for ilon, lon in enumerate(self.gcm_lon):
            for ilat, lat in enumerate(self.gcm_lat):
                interped_value  = interp_gcm_X(lon,lat,self.gcm_p,
                        self.gcm_lon,self.gcm_lat,self.gcm_p,self.X,0)
                grid_value = self.X[ilon,ilat]
                difference = interped_value - grid_value
                for idff, diff in enumerate(difference):
                    self.assertEqual(diff,0)

    def test_interpolate_at_boundary(self):
        S_pole = np.array(
            [2500.2  , 2319.8  , 2266.55 , 2254.6  , 2204.1  , 2102.7  ,
            1959.6  , 1816.75 , 1701.75 , 1613.65 , 1542.6  , 1483.35 ,
            1433.5  , 1391.65 , 1355.65 , 1323.4  , 1295.7  , 1274.65 ,
            1260.7  , 1250.65 , 1243.2  , 1234.4  , 1223.5  , 1212.4  ,
            1201.45 , 1188.7  , 1172.2  , 1155.5  , 1141.5  , 1130.35 ,
            1122.45 , 1117.   , 1111.85 , 1105.5  , 1098.25 , 1089.1  ,
            1078.   , 1066.3  , 1055.1  , 1044.65 , 1034.5  , 1023.65 ,
            1011.6  ,  998.575,  985.515,  973.71 ,  964.68 ,  959.365,
            959.055,  964.055,  976.2  ,  991.52 , 1014.55 ])
        N_pole = np.array(
            [2500.2  , 2329.3  , 2293.45 , 2291.95 , 2244.75 , 2136.4  ,
            1993.2  , 1869.2  , 1776.6  , 1703.35 , 1641.05 , 1584.75 ,
            1533.25 , 1487.55 , 1446.8  , 1408.2  , 1370.5  , 1335.5  ,
            1304.1  , 1276.85 , 1253.35 , 1233.05 , 1214.7  , 1197.9  ,
            1185.7  , 1174.55 , 1160.35 , 1144.45 , 1129.6  , 1116.5  ,
            1106.7  , 1100.45 , 1095.9  , 1091.4  , 1086.   , 1078.4  ,
            1068.3  , 1057.2  , 1046.1  , 1036.   , 1026.2  , 1015.9  ,
            1004.4  ,  991.775,  979.01 ,  967.41 ,  958.51 ,  953.095,
            952.78 ,  957.735,  970.265,  986.025, 1010.   ])
        anti_stellar = np.array(
            [2495.9   , 2260.025 , 2090.475 , 2156.05  , 2360.7   , 2476.875 ,
            2486.825 , 2430.475 , 2325.525 , 2191.5   , 2064.675 , 1961.1   ,
            1876.975 , 1806.575 , 1741.35  , 1673.675 , 1602.725 , 1532.35  ,
            1463.825 , 1394.625 , 1325.9   , 1261.025 , 1202.575 , 1152.15  ,
            1108.3   , 1067.875 , 1028.4   ,  992.0675,  960.075 ,  931.1125,
            904.1   ,  878.8275,  854.9575,  831.09  ,  807.135 ,  784.2625,
            762.825 ,  743.435 ,  726.5875,  711.7475,  698.065 ,  684.5925,
            670.245 ,  654.8275,  639.5425,  625.655 ,  613.51  ,  603.805 ,
            596.495 ,  590.99  ,  587.1725,  584.5175,  582.7375])

        south_pole = interp_gcm_X(0.,-90.,self.gcm_p,self.gcm_lon,
                self.gcm_lat,self.gcm_p,self.X,0)
        for index, value in enumerate(south_pole):
            self.assertAlmostEqual(value, S_pole[index],places=2)
        north_pole = interp_gcm_X(0,90,self.gcm_p,self.gcm_lon,
                self.gcm_lat,self.gcm_p,self.X,0)
        for index, value in enumerate(north_pole):
            self.assertAlmostEqual(value, N_pole[index],places=2)
        anti_stellar1 = interp_gcm_X(180,0,self.gcm_p,self.gcm_lon,
                self.gcm_lat,self.gcm_p,self.X,0)
        for index, value in enumerate(anti_stellar1):
            self.assertAlmostEqual(value, anti_stellar[index],places=2)
        anti_stellar2 = interp_gcm_X(-180,0,self.gcm_p,self.gcm_lon,
                self.gcm_lat,self.gcm_p,self.X,0)
        for index, value in enumerate(anti_stellar2):
            self.assertAlmostEqual(value, anti_stellar[index],places=2)

class TestInterpolateGCM(unittest.TestCase):
    gcm_lon = xlon
    gcm_lat = xlat
    gcm_p = pv
    gcm_t = tmap
    gcm_vmr = vmrmap

    def test_interpolate_out_of_bounds(self):
        with self.assertRaises(AssertionError):
            over_north_pole =  interp_gcm(0,91,self.gcm_p,self.gcm_lon,
                self.gcm_lat,self.gcm_p,self.gcm_t,self.gcm_vmr,0)
        with self.assertRaises(AssertionError):
            under_south_pole =  interp_gcm(0,-91,self.gcm_p,self.gcm_lon,
                self.gcm_lat,self.gcm_p,self.gcm_t,self.gcm_vmr,0)

    def test_interpolate_at_grid_points(self):
        for ilon, lon in enumerate(self.gcm_lon):
            for ilat, lat in enumerate(self.gcm_lat):
                interped_t,interped_vmr  = interp_gcm(lon,lat,self.gcm_p,
                        self.gcm_lon,self.gcm_lat,self.gcm_p,
                        self.gcm_t,self.gcm_vmr,0)
                grid_t = self.gcm_t[ilon,ilat,:]
                grid_vmr = self.gcm_vmr[ilon,ilat,:,:]
                for ip, p in enumerate(self.gcm_p):
                    self.assertEqual(interped_t[ip],grid_t[ip])
                    # print(interped_t[ip],grid_t[ip])
                    for ivmr,vmr in enumerate(grid_vmr[ip,:]):
                        self.assertEqual(interped_vmr[ip,ivmr],grid_vmr[ip,ivmr])
                        # print(interped_vmr[ip,ivmr],grid_vmr[ip,ivmr])

    def test_interpolate_at_boundary(self):
        S_pole_h2o = np.array(
            [0.00055616, 0.00055259, 0.00053432, 0.0005224 , 0.00051298,
            0.00050281, 0.00051121, 0.00048284, 0.00042756, 0.00041918,
            0.00042238, 0.000412  , 0.00040771, 0.00040715, 0.00040249,
            0.0003886 , 0.00038703, 0.00038419, 0.00037132, 0.00036555,
            0.0003622 , 0.0003581 , 0.00035596, 0.00035547, 0.00035481,
            0.00035424, 0.00035424, 0.000354  , 0.00035367, 0.00035359,
            0.00035359, 0.00035351, 0.00035351, 0.00035351, 0.00035351,
            0.00035351, 0.00035343, 0.00035335, 0.00035335, 0.00035327,
            0.00035318, 0.0003531 , 0.00035302, 0.00035286, 0.00035286,
            0.00035286, 0.00035286, 0.00035286, 0.00035286, 0.00035286,
            0.00035294, 0.00035294, 0.0003531 ])
        N_pole_h2o = np.array(
            [0.00055616, 0.00055094, 0.00052857, 0.00051671, 0.00050921,
            0.00049969, 0.00050641, 0.00049848, 0.00046004, 0.00042286,
            0.0004094 , 0.00038837, 0.00037879, 0.00037579, 0.00037368,
            0.0003648 , 0.00036855, 0.00036783, 0.00036129, 0.00036137,
            0.00036116, 0.00035818, 0.00035612, 0.00035588, 0.00035563,
            0.00035441, 0.00035432, 0.00035408, 0.00035367, 0.00035359,
            0.00035351, 0.00035343, 0.00035343, 0.00035343, 0.00035343,
            0.00035343, 0.00035335, 0.00035335, 0.00035327, 0.00035318,
            0.0003531 , 0.00035302, 0.00035294, 0.00035286, 0.00035286,
            0.00035286, 0.00035286, 0.00035286, 0.00035286, 0.00035286,
            0.00035286, 0.00035294, 0.0003531 ])
        anti_stellar_h2o = np.array(
            [0.00055693, 0.00056709, 0.00058271, 0.00053765, 0.00050101,
            0.00048944, 0.00048815, 0.00048778, 0.00048775, 0.00048843,
            0.00048921, 0.00048944, 0.00048921, 0.0004895 , 0.00047457,
            0.00045634, 0.00040215, 0.00037612, 0.0003606 , 0.00035553,
            0.0003561 , 0.00035686, 0.00035651, 0.00036064, 0.00035992,
            0.0003617 , 0.00036486, 0.00036825, 0.00037332, 0.00037701,
            0.00037537, 0.000389  , 0.00039875, 0.00040111, 0.00039021,
            0.00040614, 0.00043036, 0.00043043, 0.00041042, 0.00043138,
            0.00044952, 0.00045804, 0.00046172, 0.0004529 , 0.00045568,
            0.00047936, 0.00050718, 0.00051852, 0.00051406, 0.00048903,
            0.00045481, 0.00041714, 0.00039348])

        south_pole_t,south_pole_vmr = interp_gcm(0.,-90.,self.gcm_p,self.gcm_lon,
                self.gcm_lat,self.gcm_p,self.gcm_t,self.gcm_vmr,0)
        south_pole_h2o = south_pole_vmr[:,0]
        for index, value in enumerate(south_pole_h2o):
            self.assertAlmostEqual(value, S_pole_h2o[index],places=6)

        north_pole_t,north_pole_vmr = interp_gcm(0.,90.,self.gcm_p,self.gcm_lon,
                self.gcm_lat,self.gcm_p,self.gcm_t,self.gcm_vmr,0)
        north_pole_h2o = north_pole_vmr[:,0]
        for index, value in enumerate(north_pole_h2o):
            self.assertAlmostEqual(value, N_pole_h2o[index],places=6)

        anti_stellar1_pole_t,anti_stellar1_pole_vmr = interp_gcm(180,0,self.gcm_p,self.gcm_lon,
                self.gcm_lat,self.gcm_p,self.gcm_t,self.gcm_vmr,0)
        anti_stellar1_pole_h2o = anti_stellar1_pole_vmr[:,0]
        for index, value in enumerate(anti_stellar1_pole_h2o):
            self.assertAlmostEqual(value, anti_stellar_h2o[index],places=6)

        anti_stellar2_pole_t,anti_stellar2_pole_vmr = interp_gcm(-180,0,self.gcm_p,self.gcm_lon,
                self.gcm_lat,self.gcm_p,self.gcm_t,self.gcm_vmr,0)
        anti_stellar2_pole_h2o = anti_stellar2_pole_vmr[:,0]
        for index, value in enumerate(anti_stellar2_pole_h2o):
            self.assertAlmostEqual(value, anti_stellar_h2o[index],places=6)

if __name__ == "__main__":
    unittest.main()

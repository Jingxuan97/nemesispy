"""
Watch out for units...
Pressure UNIT is ATM in nemesis, which is 101325 Pa
Need to make sure pressure grid of k table is in the same unit as the calculation
unit.
"""
"""
Watch out for the stellar spectrum unit.
In .sol file, the data Stellar flux at planet's distance (W cm-2 um-1 or W cm-2 (cm-1)-1)
no need to rework using the radius.
"""
"""
R_star = 0.6668*R_SUN
planet_radius = 1*R_JUP
R_plt = 1.036*R_JUP_E
"""

aeriel_files = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/H2O_Katy_ARIEL_test',
          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CO2_Katy_ARIEL_test',
          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CO_Katy_ARIEL_test',
          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CH4_Katy_ARIEL_test']

hires_files = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/H2O_Katy_R1000',
          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CO2_Katy_R1000',
          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CO_Katy_R1000',
          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CH4_Katy_R1000']



"""
# Interpolate k lists to layers
k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)

# Mix gas opacities
k_w_g_l = new_k_overlap(k_gas_w_g_l,del_g,f)
"""



"""
         0.000  0.19739E+02    2294.2300  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
        90.390  0.11720E+02    2275.6741  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       179.552  0.69594E+01    2221.3721  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       265.916  0.41324E+01    2124.3049  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       347.980  0.24538E+01    1995.6700  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       424.832  0.14570E+01    1854.4310  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       496.285  0.86515E+00    1718.0520  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       562.745  0.51372E+00    1598.6730  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       624.995  0.30504E+00    1502.5710  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       683.970  0.18113E+00    1430.7090  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       740.575  0.10755E+00    1380.3280  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       795.573  0.63862E-01    1346.8170  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       849.542  0.37920E-01    1325.3910  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       902.885  0.22517E-01    1312.0670  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       955.869  0.13370E-01    1303.9330  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
      1008.664  0.79390E-02    1299.0250  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
      1061.373  0.47140E-02    1296.0840  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
      1114.062  0.27991E-02    1294.3311  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
      1166.766  0.16621E-02    1293.2880  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
      1219.512  0.98692E-03    1292.6680  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
"""

"""
filenames : list
    A list of strings containing names of the kta files to be read.
"""
# P_layer = np.array([])
# """
# P_layer : ndarray
#     Atmospheric pressure grid.
# """
# T_layer = np.array([])
# """
# T_layer : ndarray
#     Atmospheric temperature grid.
# """
# U_layer = np.array([])
# """
# U_layer : ndarray
#     Total number of gas particles in each layer.
# """
# f = np.array([[],[]])
# VMR_layer = f.T
# """
# f(ngas,nlayer) : ndarray
#     fraction of the different gases at each of the p-T points
# """

"""
wave_grid : ndarray
    Wavelengths (um) grid for calculating spectra.
"""
### Calling sequence
# Get averaged layer properties
"""
H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,scale,del_S\
    = average(planet_radius, H_atm, P_atm, T_atm, VMR_atm, ID, H_base)
"""

"""
 original name of this file: wasp43b.drv
      1.143   0.209844        17   0.000     :Vmin dV Npts FWHM-CORRK
    0.000   0.000       : Additional codes PAR1 and PAR2
  WASP43B.KLS
   24   0     1   0             : spectral model code, FLAGH2P, NCONT, FLAGC
 wasp43b.xsc                    : Dust x-section file
   20   1   6           : number of layers, paths and gases
    1                                    : identifier for gas 1
      0   0                              : isotope ID and process parameter
    2                                    : identifier for gas 2
      0   0                              : isotope ID and process parameter
    5                                    : identifier for gas 3
      0   0                              : isotope ID and process parameter
    6                                    : identifier for gas 4
      0   0                              : isotope ID and process parameter
   40                                    : identifier for gas 5
      0   0                              : isotope ID and process parameter
   39                                    : identifier for gas 6
      0   0                              : isotope ID and process parameter
format of layer data
 layer baseH  delH   baseP      baseT   totam       pressure    temp   doppler
        absorber amounts and partial pressures
        continuum points if any
  1    0.00   86.89 0.19739E+02 2294.230 0.44339E+27 0.16192E+02 2286.022  0.0000
         0.44339E+22 0.16192E-03 0.44339E+07 0.16192E-18 0.44339E+07 0.16192E-18
 0.44339E+07 0.16192E-18 0.66508E+26 0.24287E+01 0.37687E+27 0.13763E+02
         0.00000E+00
  2   86.89   85.68 0.12030E+02 2276.392 0.26863E+27 0.97958E+01 2253.655  0.0000
         0.26863E+22 0.97958E-04 0.26863E+07 0.97958E-19 0.26863E+07 0.97958E-19
 0.26863E+07 0.97958E-19 0.40295E+26 0.14694E+01 0.22834E+27 0.83263E+01
         0.00000E+00
  3  172.57   83.07 0.73320E+01 2225.622 0.16284E+27 0.59331E+01 2185.697  0.0000
         0.16284E+22 0.59331E-04 0.16284E+07 0.59331E-19 0.16284E+07 0.59331E-19
 0.16284E+07 0.59331E-19 0.24425E+26 0.88997E+00 0.13841E+27 0.50431E+01
         0.00000E+00
  4  255.65   79.15 0.44686E+01 2135.848 0.98846E+26 0.35983E+01 2082.271  0.0000
         0.98846E+21 0.35983E-04 0.98846E+06 0.35983E-19 0.98846E+06 0.35983E-19
 0.98846E+06 0.35983E-19 0.14827E+26 0.53975E+00 0.84018E+26 0.30585E+01
         0.00000E+00
  5  334.80   74.40 0.27234E+01 2016.333 0.60088E+26 0.21850E+01 1955.735  0.0000
         0.60088E+21 0.21850E-04 0.60088E+06 0.21850E-19 0.60088E+06 0.21850E-19
 0.60088E+06 0.21850E-19 0.90131E+25 0.32775E+00 0.51074E+26 0.18573E+01
         0.00000E+00
  6  409.19   69.41 0.16598E+01 1883.171 0.36566E+26 0.13280E+01 1822.310  0.0000
         0.36566E+21 0.13280E-04 0.36566E+06 0.13280E-19 0.36566E+06 0.13280E-19
 0.36566E+06 0.13280E-19 0.54849E+25 0.19920E+00 0.31081E+26 0.11288E+01
         0.00000E+00
  7  478.60   64.70 0.10116E+01 1751.800 0.22266E+26 0.80773E+00 1695.965  0.0000
         0.22266E+21 0.80773E-05 0.22266E+06 0.80773E-20 0.22266E+06 0.80773E-20
 0.22266E+06 0.80773E-20 0.33400E+25 0.12116E+00 0.18926E+26 0.68656E+00
         0.00000E+00
  8  543.30   60.60 0.61654E+00 1633.600 0.13562E+26 0.49153E+00 1586.105  0.0000
         0.13562E+21 0.49153E-05 0.13562E+06 0.49153E-20 0.13562E+06 0.49153E-20
 0.13562E+06 0.49153E-20 0.20343E+25 0.73729E-01 0.11527E+26 0.41779E+00
         0.00000E+00
  9  603.90   57.28 0.37576E+00 1535.138 0.82604E+25 0.29926E+00 1497.387  0.0000
         0.82604E+20 0.29926E-05 0.82604E+05 0.29926E-20 0.82604E+05 0.29926E-20
 0.82604E+05 0.29926E-20 0.12391E+25 0.44888E-01 0.70212E+25 0.25436E+00
         0.00000E+00
 10  661.18   54.76 0.22901E+00 1458.478 0.50314E+25 0.18230E+00 1430.286  0.0000
         0.50314E+20 0.18230E-05 0.50314E+05 0.18230E-20 0.50314E+05 0.18230E-20
 0.50314E+05 0.18230E-20 0.75471E+24 0.27345E-01 0.42767E+25 0.15496E+00
         0.00000E+00
 11  715.94   52.94 0.13957E+00 1402.255 0.30652E+25 0.11113E+00 1382.353  0.0000
         0.30652E+20 0.11113E-05 0.30652E+05 0.11113E-20 0.30652E+05 0.11113E-20
 0.30652E+05 0.11113E-20 0.45978E+24 0.16669E-01 0.26054E+25 0.94457E-01
         0.00000E+00
 12  768.88   51.69 0.85065E-01 1363.081 0.18683E+25 0.67788E-01 1349.684  0.0000
         0.18683E+20 0.67788E-06 0.18683E+05 0.67788E-21 0.18683E+05 0.67788E-21
 0.18683E+05 0.67788E-21 0.28025E+24 0.10168E-01 0.15881E+25 0.57619E-01
         0.00000E+00
 13  820.57   50.86 0.51844E-01 1336.891 0.11396E+25 0.41383E-01 1328.206  0.0000
         0.11396E+20 0.41383E-06 0.11396E+05 0.41383E-21 0.11396E+05 0.41383E-21
 0.11396E+05 0.41383E-21 0.17094E+24 0.62074E-02 0.96865E+24 0.35175E-01
         0.00000E+00
 14  871.44   50.33 0.31597E-01 1319.922 0.69568E+24 0.25281E-01 1314.446  0.0000
         0.69568E+19 0.25281E-06 0.69568E+04 0.25281E-21 0.69568E+04 0.25281E-21
 0.69568E+04 0.25281E-21 0.10435E+24 0.37922E-02 0.59132E+24 0.21489E-01
         0.00000E+00
 15  921.77   49.98 0.19257E-01 1309.168 0.42496E+24 0.15453E-01 1305.782  0.0000
         0.42496E+19 0.15453E-06 0.42496E+04 0.15453E-21 0.42496E+04 0.15453E-21
 0.42496E+04 0.15453E-21 0.63744E+23 0.23180E-02 0.36121E+24 0.13135E-01
         0.00000E+00
 16  971.75   49.76 0.11737E-01 1302.457 0.25983E+24 0.94514E-02 1300.391  0.0000
         0.25983E+19 0.94514E-07 0.25983E+04 0.94514E-22 0.25983E+04 0.94514E-22
 0.25983E+04 0.94514E-22 0.38974E+23 0.14177E-02 0.22085E+24 0.80336E-02
         0.00000E+00
 17 1021.51   49.62 0.71530E-02 1298.308 0.15897E+24 0.57837E-02 1297.059  0.0000
         0.15897E+19 0.57837E-07 0.15897E+04 0.57837E-22 0.15897E+04 0.57837E-22
 0.15897E+04 0.57837E-22 0.23845E+23 0.86756E-03 0.13512E+24 0.49161E-02
         0.00000E+00
 18 1071.13   49.52 0.43595E-02 1295.760 0.97338E+23 0.35409E-02 1295.010  0.0000
         0.97338E+18 0.35409E-07 0.97338E+03 0.35409E-22 0.97338E+03 0.35409E-22
 0.97338E+03 0.35409E-22 0.14601E+23 0.53113E-03 0.82736E+23 0.30097E-02
         0.00000E+00
 19 1120.65   49.46 0.26570E-02 1294.201 0.59639E+23 0.21686E-02 1293.752  0.0000
         0.59639E+18 0.21686E-07 0.59639E+03 0.21686E-22 0.59639E+03 0.21686E-22
 0.59639E+03 0.21686E-22 0.89458E+22 0.32529E-03 0.50692E+23 0.18433E-02
         0.00000E+00
 20 1170.11   49.40 0.16193E-02 1293.249 0.36563E+23 0.13286E-02 1292.982  0.0000
         0.36563E+18 0.13286E-07 0.36563E+03 0.13286E-22 0.36563E+03 0.13286E-22
 0.36563E+03 0.13286E-22 0.54845E+22 0.19930E-03 0.31078E+23 0.11293E-02
         0.00000E+00
   20   3 0.10000E-01            : Nlayers, model & error limit, path  1
   1   20 1292.982  0.10000E+01  :     layer or path, emission temp, scale
   2   19 1293.752  0.10001E+01  :     layer or path, emission temp, scale
   3   18 1295.010  0.10000E+01  :     layer or path, emission temp, scale
   4   17 1297.059  0.10000E+01  :     layer or path, emission temp, scale
   5   16 1300.391  0.99991E+00  :     layer or path, emission temp, scale
   6   15 1305.782  0.10001E+01  :     layer or path, emission temp, scale
   7   14 1314.446  0.10000E+01  :     layer or path, emission temp, scale
   8   13 1328.206  0.99990E+00  :     layer or path, emission temp, scale
   9   12 1349.684  0.10000E+01  :     layer or path, emission temp, scale
  10   11 1382.353  0.10001E+01  :     layer or path, emission temp, scale
  11   10 1430.286  0.10000E+01  :     layer or path, emission temp, scale
  12    9 1497.387  0.99999E+00  :     layer or path, emission temp, scale
  13    8 1586.105  0.10000E+01  :     layer or path, emission temp, scale
  14    7 1695.965  0.99997E+00  :     layer or path, emission temp, scale
  15    6 1822.310  0.99995E+00  :     layer or path, emission temp, scale
  16    5 1955.735  0.10000E+01  :     layer or path, emission temp, scale
  17    4 2082.271  0.99995E+00  :     layer or path, emission temp, scale
  18    3 2185.697  0.10001E+01  :     layer or path, emission temp, scale
  19    2 2253.655  0.99999E+00  :     layer or path, emission temp, scale
  20    1 2286.022  0.99997E+00  :     layer or path, emission temp, scale
    1                                    : number of filter profile points
  0.00000E+00     0.000                  : filter profile point   1
 wasp43b.out
   1                                     :number of calculations
    2   2   2   0                        :type and # of parameters for calc  1
           1
           1
   0.00000000
   0.00000000
"""
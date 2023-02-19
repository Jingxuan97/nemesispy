import unittest
import numpy as np
import scipy.interpolate
from nemesispy.common.calc_hydrostat import calc_hydrostat, calc_grav_simple
from nemesispy.common.constants import R_JUP, M_JUP, AMU, BAR, G, K_B

"""
dP/dz = -(mg/kT)*P
"""
NLAYER = 20
P1 = np.geomspace(10*BAR,1e-3*BAR,NLAYER)
T1 = np.ones(NLAYER) * 1000
mmw = np.ones(NLAYER) * 2 * AMU
M_plt = 1 * M_JUP
R_plt = 1 * R_JUP

adjusted_H = calc_hydrostat(
    P = P1,
    T = T1,
    mmw = mmw,
    M_plt = M_plt,
    R_plt = R_plt
)

print(adjusted_H)


z = (K_B*T1) / ( mmw *calc_grav_simple(adjusted_H,M_plt,R_plt))  * np.log(P1[0]/P1)
print(z)

print((adjusted_H[1:] - z[1:])/z[1:])

if __name__ == "__main__":
    unittest.main()
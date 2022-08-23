#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Calculate mean molecular weight
"""
from nemesispy.common.info_mol import mol_info
from nemesispy.common.constants import AMU

def calc_mmw(ID, VMR, ISO=[]):
    """
    Calculate mean molecular weight in kg given a list of molecule IDs and
    a list of their respective volume mixing ratios.

    Parameters
    ----------
    ID : ndarray or list
        A list of Radtran gas identifiers.
    VMR : ndarray or list
        A list of VMRs corresponding to the gases in ID.
    ISO : ndarray or list
        If ISO = [] then terrestrial relative isotopic abundance is assumed
        for all gases.
        Otherwise, if ISO[i] = 0 then terrestrial relative isotopic abundance
        is assumed for the ith gas. To specify particular isotopologue, input
        the corresponding Radtran isotopologue identifiers.

    Returns
    -------
    mmw : real
        Mean molecular weight.
        Unit: kg

    Notes
    -----
    Cf mol_id.py and mol_info.py.
    """
    mmw = 0
    if len(ISO) == 0:
        for gas_index, gas_id in enumerate(ID):
            mmw += mol_info['{}'.format(gas_id)]['mmw']*VMR[gas_index]
    else:
        for gas_index, gas_id in enumerate(ID):
            if ISO[gas_index] == 0:
                mmw += mol_info['{}'.format(gas_id)]['mmw']*VMR[gas_index]
            else:
                mmw += mol_info['{}'.format(gas_id)]['isotope']\
                    ['{}'.format(ISO[gas_index])]['mass']*VMR[gas_index]
    mmw *= AMU # keep in SI unit
    return mmw

"""
import time
start = time.time()
for i in range(2000000):
    mmw = calc_mmw([1,2],VMR=[0.5,0.5],ISO=[1,1])
    # print('mmw',mmw)
end = time.time()
print('runtime1',end-start)

start = time.time()
for i in range(2000000):
    mmw = calc_mmw([1,2],VMR=[0.5,0.5],ISO=[0,0])
    # print('mmw',mmw)
end = time.time()
print('runtime2',end-start)

start = time.time()
for i in range(2000000):
    mmw = calc_mmw([1,2],VMR=[0.5,0.5])
    # print('mmw',mmw)
end = time.time()
print('runtime3',end-start)
"""

"""
import numpy as np
mmw = calc_mmw([1,2],VMR=[0.5,0.5],ISO=[1,1])
print('mmw',mmw)
mmw = calc_mmw([1,2],VMR=[0.5,0.5],ISO=[0,0])
print('mmw',mmw)
mmw = calc_mmw([1,2],VMR=[0.5,0.5])
print('mmw',mmw)

mmw = calc_mmw(np.array([1,2]),VMR=np.array([0.5,0.5]),ISO=np.array([1,1]))
print('mmw',mmw)
mmw = calc_mmw(np.array([1,2]),VMR=np.array([0.5,0.5]),ISO=np.array([0,0]))
print('mmw',mmw)
mmw = calc_mmw(np.array([1,2]),VMR=np.array([0.5,0.5]))
print('mmw',mmw)
"""

"""Test
# ID = [1,2,3]
# ISO = None
# VMR = [0.1,0.1,0.8]
# mmw = calc_mmw(ID,VMR,ISO)
ID = [1,2,3]
ISO = [1,1,1]
VMR = [0.1,0.1,0.8]
mmw = calc_mmw(ID,VMR,ISO)
ID = [1,2,3]
ISO = None
VMR = [0.1,0.1,0.8]
mmw = calc_mmw(ID,VMR,ISO)
"""
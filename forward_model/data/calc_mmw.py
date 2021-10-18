#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

from .mol_info import mol_info

def calc_mmw(ID, VMR, ISO=None):
    """
    Calculate mean molecular weight using the information in
    Reference/mol_info.py. Molecules are referenced by their Radtran ID
    specified in Reference/radtran_id.py. By default, terrestrial
    elative isotopic abundance is assumed.

    Inputs
    ------
    ID: array,
        List of gases specified by their Radtran identifiers.
    VMR: array,
        Corresponding VMR of the gases.
    ISO: array,
        If ISO = None then terrestrial relative isotopic abundance is assumed.
        If you want to specify particular isotopes, input the Radtran isotope
        identifiers here (see ref_id.py).

    Returns
    -------
    MMW: real,
        Mean molecular weight.
    """
    NGAS = len(ID)
    MMW = 0
    for i in range(NGAS):
        if ISO == None:
            MASS = mol_info['{}'.format(ID[i])]['mmw']
        else:
            MASS = mol_info['{}'.format(ID[i])]['isotope']\
                ['{}'.format(ISO[i])]['mass']
        MMW += VMR[i] * MASS
    return MMW

ID = [1,2,3]
ISO = None
VMR = [0.1,0.1,0.8]
mmw = calc_mmw(ID,VMR,ISO)

"""Test
ID = [1,2,3]
ISO = [1,1,1]
VMR = [0.1,0.1,0.8]
mmw = calc_mmw(ID,VMR,ISO)

ID = [1,2,3]
ISO = None
VMR = [0.1,0.1,0.8]
mmw = calc_mmw(ID,VMR,ISO)
"""
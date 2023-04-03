import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

def interpolate_stellar_spectrum(output_wave, wave_data, spec_data):
    """
    Interpolate a stellar spectrum to given output wavelengths. Note that
    output_wave and wave_data must be either both in wavelengths or both in
    wavenumber.

    Parameters
    ----------
    output_wave : ndarray
        Output wavelengths/wavenumber to interpolate the stellar spectrum to.
    wave_data : ndarray
        Wavelength/wavenumber on which the stellar spectrum data is defined.
    spec_data : ndarray
        Stellar spectrum data.

    Returns
    -------
    interped_spec : ndarray
       Stellar spectrum interpolated to given wavelengths/wavenumbers
    """
    f = interpolate.interp1d(wave_data,spec_data)
    interped_spec = f(output_wave)
    return interped_spec
import numpy as np
def find_nearest(input_array, target_value):
    """
    Find the closest value in an array

    Parameters
    ----------
    input_array : ndarray/list
        An array of numbers.
    target_value : real
        Value to search for


    Returns
    -------
    idx : ndarray
        Index of closest_value within array
    array[idx] : ndarray
        Closest number to target_value in the input array
    """
    array = np.asarray(input_array)
    idx = (np.abs(array - target_value)).argmin()
    return array[idx], idx

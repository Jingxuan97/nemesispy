import numpy as np

def read_stats(folderpath, n_params, skiprows=4, filename='fix-stats.dat',
    MAP=False):
    if not MAP:
        index, means, sigma = np.loadtxt(
            '{}/{}'.format(folderpath,filename),
            skiprows=skiprows,
            unpack=True,
            max_rows=n_params
        )
    else:
        index, means = np.loadtxt(
            '{}/{}'.format(folderpath,filename),
            skiprows=skiprows+n_params+3,
            unpack=True,
            max_rows=n_params
        )
        sigma = None
    return means,sigma

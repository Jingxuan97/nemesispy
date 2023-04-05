import numpy as np

def read_stats(folderpath, n_params, skiprows=4, filename='fix-stats.dat'):
    index, means, sigma = np.loadtxt(
        '{}/{}'.format(folderpath,filename),
        skiprows=skiprows,
        unpack=True,
        max_rows=n_params
    )
    return means,sigma

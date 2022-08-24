#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Interface class for running retrieval models.
"""
import sys
import os
import numpy as np
params = {
    "1" : {
        "name" : "log_k_IR",
        "range" : (-5,3),
        "range_type" : "log",
    }
}
nparams = len(params)

try:
    import pymultinest
except ImportError:
    sys.exit("ImportError : pymultinest module")

# try:
#     from mpi4py import MPI
# except ImportError:
#     sys.exit("ImportError : pymultinest module")

bounds = np.zeros((nparams,2))
for iparam in range(nparams):

    bounds[iparam,0] = params['{}'.format(iparam)]['range'][0]
    bounds[iparam,1] = params['{}'.format(iparam)]['range'][1]

# for iparam in range(nparams):
#     if params['{}'.format(iparam)]['range_type'] == 'linear':
#         bounds[iparam,0] = params['{}'.format(iparam)]['range'][0]
#         bounds[iparam,1] = params['{}'.format(iparam)]['range'][1]
#     elif params['{}'.format(iparam)]['range_type'] == 'log':
#         bounds[iparam,0] = 10**params['{}'.format(iparam)]['range'][0]
#         bounds[iparam,1] = 10**params['{}'.format(iparam)]['range'][1
Ndata = Nphase * Nwave


class RetrievalModel():

    def __init__(self):
        pass

    def set_atmosphere_model(self):
        pass

    def set_forward_model(self):
        pass

    def run_forward_model(self, sample):
        """Output a spectrum
        """
        return model

    def run(self, params):
        Npar = len(params)
        spec = self.spec
        yerr = self.yerr

        def Prior(cube, ndim, nparams):
            for iparam in range(nparams):
                if params['{}'.format(iparam)]['range_type'] == 'linear':
                    cube[iparam] = bounds[iparam,0] \
                        + (bounds[iparam,1] - bounds[iparam,1])*cube[iparam]
                if params['{}'.format(iparam)]['range_type'] == 'log':
                    cube[iparam] = bounds[iparam,0] \
                        + (bounds[iparam,1] - bounds[iparam,1])*cube[iparam]

        def LogLikelihood(cube, ndim, nparams):
            sample = {}
            for iparam in range(nparams):
                if params['{}'.format(iparam)]['range_type'] == 'linear':
                    sample['{}'.format(iparam)] = cube[iparam]
                elif params['{}'.format(iparam)]['range_type'] == 'log':
                    sample['{}'.format(iparam)] = 10**np.array(cube[iparam])

            model = self.run_forward_model(sample)
            loglikelihood= -0.5*(np.sum((spec-model)**2/yerr**2))
            return loglikelihood

        pymultinest.run(
            LogLikelihood = LogLikelihood,
            Prior = Prior,
            n_dims = Npar,
            n_live_points = 400,
            sampling_efficiency = 0.8,
            outputfiles_basename = "chains/1-",
        )

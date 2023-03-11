This library contains routines for simulating and fitting
exoplanet emission spectra at arbitary orbital phase,
thereby constraining the thermal structure and chemical
abundance of exoplanet atmospheres. It is also capable
of fitting emission spectra at multiple orbital phases
(phase curves) at the same time. This package
comes ready with some spectral data and general circulation
model data so you could start simulating spectra
straight away. There are a few demonstration routines in
the `nemesispy` folder; in particular the `demo_fit_eclipse.py `
contains an interactive plot routine which allows you to try
to fit a hot Jupiter eclipse spectrum by hand by varying
its chemical abundance and temperature profile. This package
can be easily integrated with a Bayesian sampler, in particular
`MultiNest` for a full spectral retrieval.

The radiative transfer calculations are done with the
correlated-k approximation, and are accelerated with the
`numba` just-in-time compiler to match the speed of
compiled languages such as Fortran. The radiative transfer
routines are based on the well-tested [nemesis](https://github.com/nemesiscode) library developed
by Patrick Irwin (Univeristy of Oxford) and collaborators.

This package is following features

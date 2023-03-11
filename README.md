This library contains routines for simulating and fitting
exoplanet emission spectra at arbitrary orbital phase,
thereby constraining the thermal structure and chemical
abundance of exoplanet atmospheres. It is also capable
of fitting emission spectra at multiple orbital phases
(phase curves) at the same time. This package
comes ready with some spectral data and General Circulation
Model (GCM) )data so you could start simulating spectra
straight away. There are a few demonstration routines in
the `nemesispy` folder; in particular `demo_fit_eclipse.py`
contains an interactive plot routine which allows you
to fit a hot Jupiter eclipse spectrum by hand by varying
its chemical abundance and temperature profile. This package
can be easily integrated with a Bayesian sampler, in particular
`MultiNest` for a full spectral retrieval.

The radiative transfer calculations are done with the
correlated-k approximation, and are accelerated with the
`numba` just-in-time compiler to match the speed of
compiled languages such as Fortran. The radiative transfer
routines are based on the well-tested [Nemesis](https://github.com/nemesiscode) library developed
by Patrick Irwin (University of Oxford) and collaborators.

This package has the following advantageous features:

* Highly portable and customisable compared
  to packages written in compiled languages, and
  can be easily installed on computer clusters.
* Fast calculation speed due to just-in-time
  compilation, which compiles Python code to machine
  code at run time.
* Radiative transfer routines are benchmarked against
  the extensively used [Nemesis](https://github.com/nemesiscode) library.
* Contains interactive plotting routines that allows you
  to visualise the impact of gas abundance and thermal
  structure on the emission spectra.
* Contains routine to simulate spectra from General
  Circulation Models (GCMs).
* Contains unit tests so that you could check if the
  the code is working properly after your modifications.

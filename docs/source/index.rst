.. NemesisPy documentation master file, created by
   sphinx-quickstart on Mon Mar  4 13:39:48 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NemesisPy's documentation!
=====================================

This library contains routines for simulating and fitting
exoplanet emission spectra at arbitrary orbital phase,
thereby constraining the thermal structure and chemical
abundance of exoplanet atmospheres. It is also capable
of fitting emission spectra at multiple orbital phases
(phase curves) at the same time. This package
comes ready with some spectral data and General Circulation
Model (GCM) data so you could start simulating spectra
straight away. There are a few demonstration routines in
the `nemesispy` folder; in particular `demo_fit_eclipse.py`
contains an interactive plot routine which allows you
to fit a hot Jupiter eclipse spectrum by hand by varying
its chemical abundance and temperature profile. This package
can be easily integrated with a Bayesian sampler, in particular
`MultiNest` for a full spectral retrieval.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
   usage



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

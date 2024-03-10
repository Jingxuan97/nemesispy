.. NemesisPy documentation master file, created by
   sphinx-quickstart on Sun Mar 10 22:23:59 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NemesisPy's documentation!
=====================================

**NemesisPy** contains routines for calculating and fitting
exoplanet emission spectra at arbitrary orbital phases,
which can help us constrain the thermal structure and chemical
abundance of exoplanet atmospheres. It is also capable
of fitting emission spectra at multiple orbital phases
(phase curves) at the same time. This package
comes ready with some spectral data and General Circulation
Model (GCM) data so you can start simulating spectra immediately.
There are a few demonstration routines in
the `nemesispy/examples` folder; in particular, `demo_fit_eclipse.py`
contains an interactive plot routine which allows you
to fit a hot Jupiter eclipse spectrum by hand by varying
its chemical abundance and temperature profile. This package
can be easily integrated with a Bayesian sampler such as
`MultiNest` for a full spectral retrieval.

The radiative transfer calculations are done with the
correlated-k approximation, and are accelerated with the
`Numba` just-in-time compiler to match the speed of
compiled languages such as Fortran. The radiative transfer
routines are based on the well-tested Nemesis (https://github.com/nemesiscode)
library developed by Patrick Irwin (University of Oxford) and collaborators.

This package has the following features:

* Written fully in Python: highly portable and customisable compared
  to packages written in compiled languages and
  can be easily installed on computer clusters.
* Fast calculation speed: the most time consuming routines are optimised with
  just-in-time compilation, which compiles Python code to machine
  code at run time.
* Radiative transfer routines are benchmarked against
  the extensively used Nemesis (https://github.com/nemesiscode) library.
* Contains interactive plotting routines that allow you
  to visualise the impact of gas abundance and thermal
  structure on the emission spectra.
* Contains routines to simulate spectra from General
  Circulation Models (GCMs).
* Contains unit tests to check if
  the code is working correctly after modifications.

.. code-block:: console

    $ pip install --editable .

To run all unit tests, change directory to the software folder and type the
following in the terminal:
.. code-block:: console

    $ python -m unittest discover test/

.. note::

   This project is under active development.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Contents
--------

.. toctree::

   usage
   api
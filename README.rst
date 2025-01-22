See https://jingxuan97.github.io/nemesispy/ for documentation.

============
Introudction
============

**NEMESISPY** contains routines for calculating and fitting
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
routines are based on the well-tested NEMESIS (https://github.com/nemesiscode)
library developed by Patrick Irwin (University of Oxford) and collaborators.

This package has the following features:

* Written fully in Python: highly portable and customisable compared
  to packages written in compiled languages and
  can be easily installed on computer clusters.
* Fast calculation speed: the most time consuming routines are optimised with
  just-in-time compilation, which compiles Python code to machine
  code at run time.
* Radiative transfer routines are benchmarked against
  the extensively used NEMESIS (https://github.com/nemesiscode) library.
* Contains interactive plotting routines that allow you
  to visualise the impact of gas abundance and thermal
  structure on the emission spectra.
* Contains routines to simulate spectra from General
  Circulation Models (GCMs).
* Contains unit tests to check if
  the code is working correctly after modifications.

============
Installation
============

In order to install the package but still make it editable, change directory to
the software folder and type the following in the terminal:

.. code-block:: console

    $ pip install --editable .

=====
Tests
=====

To run all unit tests, change directory to the software folder and type the
following in the terminal:

.. code-block:: console

    $ python -m unittest discover test/


=======
Contact
=======

The project is currently maintained by `Jingxuan Yang <https://scholar.google.com/citations?user=2XEkBdUAAAAJ&hl=en>`_.
If you would like to contribute to the project, please contact the maintainer.

Contributors: Jingxuan Yang, Juan Alday, Agnibha Banerjee.

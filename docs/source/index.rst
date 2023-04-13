.. nemesispy documentation master file, created by
   sphinx-quickstart on Mon Mar 27 19:30:56 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to nemesispy's documentation!
=====================================

**nemesispy** is a Python library for simulating and fitting low resolution
exoplanet spectra taken by facilities such as the Hubble Space Telescope
and the James Webb Space Telescope. **nemesispy** can generate
disc-averaged emission spectra of a transitting hot Jupiter at arbitrary
orbital phase from an atmospheric model. It contains routines to fit all
phases of a spectroscopic phase curve simulaneously. The radiative transfer
calculations are done with the fast correlated-k method, and the package
can be easily coupled to a Bayesian parameter estimation scheme for retrievals.
Additionally, the package contains interactive plotting routines to demonstrate
the basics of atmospheric retrievals, as well as visualising the
transmission weighting function.

The package can be installed from
`Python Package Index <https://pypi.org/project/nemesispy/>`_ (PyPI)
and the source code can be viewed on
`GitHub <https://github.com/Jingxuan97/nemesispy>`_.

Check out the :doc:`start`

.. .. autosummary::
..    :toctree: _autosummary
..    :template: custom-module-template.rst
..    :recursive:

..    nemesispy

.. note::

        This project is under active development.

.. toctree::
   :maxdepth: 2
   :caption: Contents:




Contents
--------

.. toctree::

   start
   api
   nemesispy


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

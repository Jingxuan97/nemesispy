---
title: 'NemesisPy: A Python package for simulating and retrieving exoplanetary spectra'
tags:
  - Python
  - astronomy
  - exoplanets
  - spectroscopy
  - radiative transfer
  - atmospheric retrieval

authors:
  - name: Jingxuan Yang
    orcid: 0009-0006-2395-6197
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Juan Alday
    orcid: 0000-0003-1459-3444
    corresponding: false # (This is how to denote the corresponding author)
    affiliation: 2
  - name: Patrick Irwin
    orcid: 0000-0002-6772-384X
    corresponding: false # (This is how to denote the corresponding author)
    affiliation: 1
affiliations:
 - name: Department of Physics, University of Oxford, Parks Road, Oxford OX1 3PU, UK
   index: 1
 - name: School of Physical Sciences, The Open University, Walton Hall, Milton Keynes MK7 6AA, UK
   index: 2
date: 13 August 2017
bibliography: paper.bib
---
# Summary

Spectra of exoplanets allow us to probe their atmospheres' composition and thermal structure and, when applicable, their surface conditions [@burrows_spectra_2014]. Spectroscopic characterisation of a large population of exoplanets may help us understand the origin and evolution of planetary systems [@mordasini_imprint_2016; @madhusudhan_atmospheric_2017; @chachan_breaking_2023]. The extraction of information from spectral data is known as atmospheric retrievals [e.g., @irwin_nemesis_2008; @madhusudhan_temperature_2009; @line_systematic_2013], which can be divided into two steps: forward modelling and model fitting. The forward modelling step requires, at a minimum, an atmospheric model for the observed planet and a radiative transfer pipeline that can calculate model spectra given an atmospheric model. The model fitting step typically requires a Bayesian parameter inference algorithm that can constrain the free parameters of the forward model by fitting the observed spectra. Atmospheric retrieval pipelines have long been applied to the spectral analysis of the Earth and other solar system planets, and the discovery of exoplanets further ignited the developments of new retrieval pipelines with varying focus and functionalities [@macdonald_catalog_2023].

NemesisPy is a Python package developed to perform parametric atmospheric modelling and radiative transfer calculation for the retrievals of exoplanetary spectra. It is a recent development of the well-established Fortran NEMESIS library [@irwin_nemesis_2008], which has been applied to the retrievals of both solar system planets and exoplanets spectra using different observing geometries [ @lee_optimal_2012; @teanby_active_2012; @barstow_clouds_2014; @krissansen-totton_detectability_2018; @barstow_unveiling_2020; @irwin_25d_2020; @james_temporal_2023]. NemesisPy can be easily interfaced with Bayesian inference algorithms to retrieve atmospheric properties from spectroscopic observations. For example, NemesisPy has been applied to the retrievals of Hubble and Spitzer data of a hot Jupiter [@yang_testing_2023], as well as to JWST/Mid-Infrared Instrument (JWST/MIRI) data of a hot Jupiter (Yang et al. 2024, submitted).

# Statement of need

NemesisPy has three distinguishing features as an exoplanetary retrieval pipeline. Firstly, NemesisPy inherits the fast correlated-k [@lacis_description_1991] radiative transfer routine from the Fortran NEMESIS library [@irwin_nemesis_2008], which has been extensively validated against other radiative transfer codes [@barstow_comparison_2020]. Secondly, NemesisPy employs a just-in-time compiler [@lam_numba_2015], which compiles the most computationally expensive routines to machine code at run time. Combined with extensive code refactoring, NemesisPy is significantly faster than the Fortran NEMESIS library. Such speed improvement is crucial for analysing exoplanetary spectra using sampling-based Bayesian parameter estimation [e.g., @feroz_multimodal_2008], which typically involves the computation of millions of model spectra. Thirdly, it implements several parametric atmospheric temperature models described in [@yang_testing_2023]. These routines are particularly useful for retrieving spectroscopic phase curves of hot Jupiters, which are emission spectra observed at multiple orbital phases and can enable detailed atmospheric characterisation.

NemesisPy contains several general-purpose routines for atmospheric modelling and spectral simulations. The modular nature of the package means that subroutines can be easily called on their own. Currently, NemesisPy has an easy-to-use API for simulating emission spectra and phase curves of hot Jupiters from arbitrary input atmospheric models, and we are currently developing an interface for retrieving transmission spectra. NemesisPy has already been used in a scientific publication [@yang_testing_2023] and is actively used in exoplanetary data analysis projects. The combination of well-tested core radiative transfer routines, accelerated computational speed, and packaged modular design is ideal for tackling the influx of JWST data of exoplanets.

# Acknowledgements

The authors express gratitude to the developers of many open-source Python packages used by NemesisPy, in particular numpy [@harris_array_2020], SciPy [@virtanen_scipy_2020], Numba [@lam_numba_2015] and Matplotlib [@hunter_matplotlib_2007]. The authors also express gratidude to the many developers of the open-source Fortran NEMESIS library [@irwin_nemesis_2008].

# References

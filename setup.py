from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "Tools for modelling exoplanet spectra"
LONG_DESCRIPTION = """Tools for analysing exoplanets emission spectra and \
phase curves based on the Fortran Nemesis library"""

setup(name="nemesispy",
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author="Jingxuan Yang",
      author_email="jingxuanyang15@gmail.com",
      packages=find_packages(),
      install_requires=[
          "numpy",
          "scipy",
          "matplotlib",
          "numba"]
        )
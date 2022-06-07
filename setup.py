from pydoc import describe
from setuptools import setup

VERSION = "0.0.1"
DESCRIPTION = "Python package to simulate exoplanet phase curves"
LONG_DESCRIPTION = """Python package to simulate disk-averaged atmospheric \
emission spectra of exoplanets at multiple orbital phases using the \
correlated-k method. The code is based on the Fortran Nemesis library."""

setup(name="nemesispy",
      version=VERSION,
      author="Jingxuan Yang",
      author_email="<jingxuanyang15@gmail.com>",
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      packages=["nemesispy"],
      install_requires=[
          "numpy",
          "numba",
          "scipy",
          "matplotlib",
          "pymultinest"])
      # install_requires=["numpy","matplotlib","sympy","miepython","numba","ray"])
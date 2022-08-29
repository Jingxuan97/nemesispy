from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "Python package to simulate exoplanet phase curves"
LONG_DESCRIPTION = """Python package to simulate disk-averaged atmospheric \
emission spectra of exoplanets at multiple orbital phases using the \
correlated-k method. The code is based on the Fortran Nemesis library."""

setup(name="nemesispy",
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author="Jingxuan Yang",
      author_email="<jingxuanyang15@gmail.com>",
      # packages=["nemesispy"],
      packages=find_packages(),
      install_requires=[
          "numpy",
          "scipy",
          "matplotlib"])
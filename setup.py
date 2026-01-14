from setuptools import setup, find_packages

with open('README.rst') as f:
    long_description = f.read()

VERSION = "0.0.10"
DESCRIPTION = "Tools for modelling exoplanet spectra"
setup(name="nemesispy",
      version=VERSION,
      description=DESCRIPTION,
      url='https://github.com/Jingxuan97/nemesispy',
      author="Jingxuan Yang",
      author_email="jingxuanyang15@gmail.com",
      packages=find_packages(),
      package_data={
          '':[
               'nemesispy/data/*.txt',
               'nemesispy/data/*/*.txt',
               'nemesispy/data/*/*/*.txt',
               'nemesispy/data/cia/*.tab',
               'nemesispy/data/ktables/*.cia',
          ]
      },
      include_package_data=True,
      install_requires=[
          "numpy>=1.19.0,<1.25",
          "scipy>=1.5.0",
          "matplotlib>=3.3.0",
          "numba>=0.51.0",
          "llvmlite==0.40.1"],
      long_description=long_description,
      long_description_content_type='text/markdown',
        )
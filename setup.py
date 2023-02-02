from setuptools import setup, find_packages

VERSION = "0.0.5"
DESCRIPTION = "Tools for modelling exoplanet spectra"
setup(name="nemesispy",
      version=VERSION,
      description=DESCRIPTION,
      url='https://github.com/Jingxuan97/nemesispy2022',
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
          "numpy",
          "scipy",
          "matplotlib",
          "numba"]
        )
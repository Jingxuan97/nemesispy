from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

VERSION = "0.0.6"
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
          "numba"],
      long_description=long_description,
      long_description_content_type='text/markdown',
        )
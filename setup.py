from setuptools import setup

setup(name='nemesispy',
      version='0.0',
      description='NEMESIS radiative transfer code',
      packages=['nemesispy'],
      install_requires=['numpy','scipy','sympy'])
      # install_requires=['numpy','matplotlib','sympy','miepython','numba','ray'])
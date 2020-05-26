# encoding: utf-8
from setuptools import setup

setup(name='platalea',
      version='0.1',
      description='Understanding visually grounded spoken language via multi-tasking',
      url='https://github.com/gchrupala/platalea',
      author='Grzegorz Chrupała',
      author_email='g.chrupala@uvt.nl',
      license='Apache',
      zip_safe=False,
      packages=["platalea", "platalea.audio"],  # , "platalea.utils", "platalea.experiments"],
      install_requires=[
          'torch>=1.2.0',
          'torchvision>=0.4.0',
          'numpy>=1.17.2',
          'scipy>=1.3.1',
          'configargparse>=1.0',
          'nltk>=3.4.5'
      ])

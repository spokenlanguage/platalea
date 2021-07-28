# encoding: utf-8
from setuptools import setup

setup(name='platalea',
      description='Understanding visually grounded spoken language via multi-tasking',
      url='https://github.com/spokenlanguage/platalea',
      author='Grzegorz Chrupała',
      author_email='g.chrupala@uvt.nl',
      license='Apache',
      zip_safe=False,
      packages=['platalea', 'platalea.audio', 'platalea.utils', 'platalea.experiments',
                'platalea.experiments.flickr8k', 'platalea.experiments.librispeech_places'],
      include_package_data=True,
      install_requires=[
          'torch==1.8.1',
          'torchvision>=0.4.0',
          'numpy>=1.17.2',
          'scipy>=1.3.1',
          'configargparse>=1.0',
          'nltk>=3.4.5',
          'soundfile>=0.10.3',
          'scikit-learn==0.22.1',
          'wandb>=0.10.10',
          'python-Levenshtein>=0.12.0'],
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      )

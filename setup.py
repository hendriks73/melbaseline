from setuptools import setup, find_packages

import glob

# define scripts to be installed by the PyPI package
scripts = glob.glob('bin/*')

setup(name='melbaseline',
      version='0.1',
      packages=find_packages(),
      description='Baseline for Mediaeval 2018 Acousticbrainz genre classification task.',
      author='Hendrik Schreiber',
      author_email='hs@tagtraum.com',
      license='Private',
      scripts=scripts,
      install_requires=[
          'h5py',
          'scikit-learn',
          'bz2file',
          'tensorflow-gpu==1.10.1;platform_system!="Darwin"',
          'tensorflow==1.10.1;platform_system=="Darwin"',
      ],
      include_package_data=True,
      zip_safe=False)

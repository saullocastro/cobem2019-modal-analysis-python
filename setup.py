from __future__ import absolute_import

import os
import inspect
import subprocess
from setuptools import setup, find_packages


is_released = True
version = '1.0'

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    setupdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    return open(os.path.join(setupdir, fname)).read()


#_____________________________________________________________________________

install_requires = [
        "numpy",
        "scipy",
        "sympy",
        "matplotlib",
        "numba",
        ]

#Trove classifiers
CLASSIFIERS = """\

Development Status :: 3 - Alpha
Intended Audience :: Education
Intended Audience :: Science/Research
Intended Audience :: Developers
Topic :: Scientific/Engineering :: Mathematics
Topic :: Education
License :: OSI Approved :: BSD License
Operating System :: Microsoft :: Windows
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Operating System :: Unix

"""

data_files = [('', [
        'README.md',
        'LICENSE',
        ])]

s = setup(
    name = "cobem2019-modal-analysis-python",
    version = version,
    author = "Saullo G. P. Castro",
    author_email = "S.G.P.Castro@tudelft.nl",
    description = (""),
    license = "BSD License 2.0",
    keywords = "structural analysis vibration dynamics finite elements",
    url = "https://github.com/saullocastro/cobem2019-modal-analysis-python",
    data_files=data_files,
    long_description=read('README.md'),
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    install_requires=install_requires,
    include_package_data=True,
)


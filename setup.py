try:
    from setuptools import setup
    from setuptools.extension import Extension
except:
    from distutils.core import setup
    from distutils.extension import Extension

import numpy as np
from Cython.Distutils import build_ext
import sys, os, re
from sys import platform
from os import environ

## https://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class build_ext_subclass( build_ext ):
    def build_extensions(self):
        c = self.compiler.compiler_type
        # TODO: add entries for intel's ICC
        if c == 'msvc': # visual studio
            for e in self.extensions:
                e.extra_compile_args = ['/openmp', '/O2', '/std:c++14']
                ### Note: MSVC never implemented C++11
        elif (c == "clang") or (c == "clang++"):
            for e in self.extensions:
                e.extra_compile_args = ['-fopenmp', '-O2', '-march=native', '-std=c++17']
                e.extra_link_args    = ['-fopenmp']
                ### Note: when passing C++11 to CLANG, it complains about C++17 features in CYTHON_FALLTHROUGH
        else: # gcc
            for e in self.extensions:
                e.extra_compile_args = ['-fopenmp', '-O2', '-march=native', '-std=c++11']
                e.extra_link_args    = ['-fopenmp']

                # e.extra_compile_args = ['-O2', '-march=native', '-std=c++11']

                ### for testing (run with `LD_PRELOAD=libasan.so python script.py`)
                # extra_compile_args=["-std=c++11", "-fsanitize=address", "-static-libasan", "-ggdb"],
                # extra_link_args = ["-fsanitize=address", "-static-libasan"]
                # e.define_macros += [("TEST_MODE_DEFINE", None)]

        ## Note: apple will by default alias 'gcc' to 'clang', and will ship its own "special"
        ## 'clang' which has no OMP support and nowadays will purposefully fail to compile when passed
        ## '-fopenmp' flags. If you are using mac, and have an OMP-capable compiler,
        ## comment out the code below, or set 'use_omp' to 'True'.
        if not use_omp:
            for e in self.extensions:
                e.extra_compile_args = [arg for arg in e.extra_compile_args if arg != '-fopenmp']
                e.extra_link_args    = [arg for arg in e.extra_link_args    if arg != '-fopenmp']

        build_ext.build_extensions(self)

use_omp = (("enable-omp" in sys.argv)
           or ("-enable-omp" in sys.argv)
           or ("--enable-omp" in sys.argv))
if use_omp:
    sys.argv = [a for a in sys.argv if a not in ("enable-omp", "-enable-omp", "--enable-omp")]
if environ.get('ENABLE_OMP') is not None:
    use_omp = True
if platform[:3] != "dar":
    use_omp = True

### Shorthand for apple computer:
### uncomment line below
# use_omp = True

setup(
    name  = "outliertree",
    packages = ["outliertree"],
    version = '1.4.0',
    description = 'Explainable outlier detection through smart decision tree conditioning',
    author = 'David Cortes',
    author_email = 'david.cortes.rivera@gmail.com',
    url = 'https://github.com/david-cortes/outliertree',
    keywords = ['outlier', 'anomaly', 'gritbot'],
    cmdclass = {'build_ext': build_ext_subclass},
    ext_modules = [Extension(
                                "outliertree._outlier_cpp_interface",
                                sources=["outliertree/outlier_cpp_interface.pyx", "src/split.cpp", "src/cat_outlier.cpp",
                                         "src/fit_model.cpp", "src/clusters.cpp", "src/misc.cpp", "src/predict.cpp"],
                                include_dirs=[np.get_include(), ".", "./src"],
                                define_macros=[("_FOR_PYTHON", None)],
                                language="c++",
                                install_requires = ["numpy", "pandas>=0.24.0", "cython"]
                            )]
    ) 

if not use_omp:
    import warnings
    apple_msg  = "\n\n\nMacOS detected. Package will be built without multi-threading capabilities, "
    apple_msg += "due to Apple's lack of OpenMP support in default clang installs. In order to enable it, "
    apple_msg += "install the package directly from GitHub: https://www.github.com/david-cortes/outliertree\n"
    apple_msg += "Using 'python setup.py install enable-omp'. "
    apple_msg += "You'll also need an OpenMP-capable compiler.\n\n\n"
    warnings.warn(apple_msg)

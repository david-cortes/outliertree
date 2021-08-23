try:
    from setuptools import setup
    from setuptools.extension import Extension
except:
    from distutils.core import setup
    from distutils.extension import Extension

import numpy as np
from Cython.Distutils import build_ext
import sys, os, subprocess, warnings, re
from sys import platform
from os import environ

found_omp = True
def set_omp_false():
    global found_omp
    found_omp = False

## https://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class build_ext_subclass( build_ext ):
    def build_extensions(self):
        is_msvc = self.compiler.compiler_type == "msvc"
        is_clang = hasattr(self.compiler, 'compiler_cxx') and ("clang++" in self.compiler.compiler_cxx)

        if not is_msvc:
            self.add_march_native()
            self.add_openmp_linkage()
            self.add_restrict_qualifier()
            if sys.platform[:3].lower() != "win":
                self.add_link_time_optimization()

        if is_msvc:
            for e in self.extensions:
                e.extra_compile_args += ['/openmp', '/O2', '/std:c++14']
                ### Note: MSVC never implemented C++11
        elif is_clang:
            for e in self.extensions:
                e.extra_compile_args += ['-O2', '-std=c++17']
                ### Note: when passing C++11 to CLANG, it complains about C++17 features in CYTHON_FALLTHROUGH
        else: # gcc
            for e in self.extensions:
                # e.extra_compile_args = ['-fopenmp', '-O2', '-march=native', '-std=c++11']
                # e.extra_link_args    = ['-fopenmp']
                e.extra_compile_args += ['-O2', '-std=c++11']

                # e.extra_compile_args = ['-O2', '-march=native', '-std=c++11']

                ### for testing (run with `LD_PRELOAD=libasan.so python script.py`)
                # extra_compile_args=["-std=c++11", "-fsanitize=address", "-static-libasan", "-ggdb"],
                # extra_link_args = ["-fsanitize=address", "-static-libasan"]
                # e.define_macros += [("TEST_MODE_DEFINE", None)]

        build_ext.build_extensions(self)

    def add_march_native(self):
        arg_march_native = "-march=native"
        arg_mcpu_native = "-mcpu=native"
        if self.test_supports_compile_arg(arg_march_native):
            for e in self.extensions:
                e.extra_compile_args.append(arg_march_native)
        elif self.test_supports_compile_arg(arg_mcpu_native):
            for e in self.extensions:
                e.extra_compile_args.append(arg_mcpu_native)

    def add_link_time_optimization(self):
        arg_lto = "-flto"
        if self.test_supports_compile_arg(arg_lto):
            for e in self.extensions:
                e.extra_compile_args.append(arg_lto)
                e.extra_link_args.append(arg_lto)

    def add_openmp_linkage(self):
        arg_omp1 = "-fopenmp"
        arg_omp2 = "-qopenmp"
        arg_omp3 = "-xopenmp"
        arg_omp4 = "-fiopenmp"
        args_apple_omp = ["-Xclang", "-fopenmp", "-lomp"]
        if self.test_supports_compile_arg(arg_omp1):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp1)
                e.extra_link_args.append(arg_omp1)
        elif (sys.platform[:3].lower() == "dar") and self.test_supports_compile_arg(args_apple_omp):
            for e in self.extensions:
                e.extra_compile_args += ["-Xclang", "-fopenmp"]
                e.extra_link_args += ["-lomp"]
        elif self.test_supports_compile_arg(arg_omp2):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp2)
                e.extra_link_args.append(arg_omp2)
        elif self.test_supports_compile_arg(arg_omp3):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp3)
                e.extra_link_args.append(arg_omp3)
        elif self.test_supports_compile_arg(arg_omp4):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp4)
                e.extra_link_args.append(arg_omp4)
        else:
            set_omp_false()

    def test_supports_compile_arg(self, comm):
        is_supported = False
        try:
            if not hasattr(self.compiler, "compiler_cxx"):
                return False
            if not isinstance(comm, list):
                comm = [comm]
            print("--- Checking compiler support for option '%s'" % " ".join(comm))
            fname = "outliertree_compiler_testing.cpp"
            with open(fname, "w") as ftest:
                ftest.write(u"int main(int argc, char**argv) {return 0;}\n")
            try:
                cmd = [self.compiler.compiler_cxx[0]]
            except:
                cmd = list(self.compiler.compiler_cxx)
            val_good = subprocess.call(cmd + [fname])
            try:
                val = subprocess.call(cmd + comm + [fname])
                is_supported = (val == val_good)
            except:
                is_supported = False
        except:
            pass
        try:
            os.remove(fname)
        except:
            pass
        return is_supported

    def add_restrict_qualifier(self):
        supports_restrict = False
        try:
            if not hasattr(self.compiler, "compiler_cxx"):
                return None
            print("--- Checking compiler support for '__restrict' qualifier")
            fname = "outliertree_compiler_testing.cpp"
            with open(fname, "w") as ftest:
                ftest.write(u"int main(int argc, char**argv) {return 0;}\n")
            try:
                cmd = [self.compiler.compiler_cxx[0]]
            except:
                cmd = list(self.compiler.compiler_cxx)
            val_good = subprocess.call(cmd + [fname])
            try:
                with open(fname, "w") as ftest:
                    ftest.write(u"int main(int argc, char**argv) {double *__restrict x = nullptr; return 0;}\n")
                val = subprocess.call(cmd + [fname])
                supports_restrict = (val == val_good)
            except:
                return None
        except:
            pass
        try:
            os.remove(fname)
        except:
            pass
        
        if supports_restrict:
            for e in self.extensions:
                e.define_macros += [("SUPPORTS_RESTRICT", "1")]


setup(
    name  = "outliertree",
    packages = ["outliertree"],
    version = '1.7.5',
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

if not found_omp:
    omp_msg  = "\n\n\nCould not detect OpenMP. Package will be built without multi-threading capabilities. "
    omp_msg += " To enable multi-threading, first install OpenMP"
    if (sys.platform[:3] == "dar"):
        omp_msg += " - for macOS: 'brew install libomp'\n"
    else:
        omp_msg += " modules for your compiler. "
    
    omp_msg += "Then reinstall this package from scratch: 'pip install --force-reinstall outliertree'.\n"
    warnings.warn(omp_msg)

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
        is_windows = sys.platform[:3] == "win"
        is_mingw = (is_windows and
                    (self.compiler.compiler_type.lower()
                     in ["mingw32", "mingw64", "mingw", "msys", "msys2", "gcc", "g++"]))

        if not is_msvc:
            if not self.check_for_variable_dont_set_march() and not self.check_cflags_or_cxxflags_contain_arch():
                self.add_march_native()
            self.add_openmp_linkage()
            self.add_restrict_qualifier()
            self.add_no_math_errno()
            self.add_no_trapping_math()
            if sys.platform[:3].lower() != "win":
                self.add_link_time_optimization()

        if is_msvc:
            for e in self.extensions:
                e.extra_compile_args += ['/openmp', '/O2', '/std:c++14', '/fp:except-', '/wd4267', '/wd4018']
                ### Note: MSVC never implemented C++11
        elif is_clang:
            for e in self.extensions:
                e.extra_compile_args += ['-O2', '-std=c++17']
                ### Note: when passing C++11 to CLANG, it complains about C++17 features in CYTHON_FALLTHROUGH
        else: # gcc
            self.add_O2()
            self.add_std_cpp11()
            for e in self.extensions:
                # e.extra_compile_args = ['-fopenmp', '-O2', '-march=native', '-std=c++11']
                # e.extra_link_args    = ['-fopenmp']

                if is_mingw:
                    e.extra_compile_args += ['-Wno-sign-compare', '-Wno-maybe-uninitialized']

                # e.extra_compile_args = ['-O2', '-march=native', '-std=c++11']

                ### for testing (run with `LD_PRELOAD=libasan.so python script.py`)
                # extra_compile_args=["-std=c++11", "-fsanitize=address", "-static-libasan", "-ggdb"],
                # extra_link_args = ["-fsanitize=address", "-static-libasan"]
                # e.define_macros += [("TEST_MODE_DEFINE", None)]

        build_ext.build_extensions(self)

    def check_cflags_or_cxxflags_contain_arch(self):
        arch_list = ["-march", "-mcpu", "-mtune", "-msse", "-msse2", "-msse3", "-mssse3", "-msse4", "-msse4a", "-msse4.1", "-msse4.2", "-mavx", "-mavx2"]
        for env_var in ("CFLAGS", "CXXFLAGS"):
            if env_var in os.environ:
                for flag in arch_list:
                    if flag in os.environ[env_var]:
                        return True
        return False

    def check_for_variable_dont_set_march(self):
        return "DONT_SET_MARCH" in os.environ

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

    def add_no_math_errno(self):
        arg_fnme = "-fno-math-errno"
        if self.test_supports_compile_arg(arg_fnme):
            for e in self.extensions:
                e.extra_compile_args.append(arg_fnme)
                e.extra_link_args.append(arg_fnme)

    def add_no_trapping_math(self):
        arg_fntm = "-fno-trapping-math"
        if self.test_supports_compile_arg(arg_fntm):
            for e in self.extensions:
                e.extra_compile_args.append(arg_fntm)
                e.extra_link_args.append(arg_fntm)

    def add_O2(self):
        arg_O2 = "-O2"
        if self.test_supports_compile_arg(arg_O2):
            for e in self.extensions:
                e.extra_compile_args.append(arg_O2)
                e.extra_link_args.append(arg_O2)

    def add_std_cpp11(self):
        arg_std_cpp11 = "-std=c++11"
        if self.test_supports_compile_arg(arg_std_cpp11):
            for e in self.extensions:
                e.extra_compile_args.append(arg_std_cpp11)
                e.extra_link_args.append(arg_std_cpp11)


    def add_openmp_linkage(self):
        arg_omp1 = "-fopenmp"
        arg_omp2 = "-qopenmp"
        arg_omp3 = "-xopenmp"
        arg_omp4 = "-fiopenmp"
        args_apple_omp = ["-Xclang", "-fopenmp", "-lomp"]
        args_apple_omp2 = ["-Xclang", "-fopenmp", "-L/usr/local/lib", "-lomp", "-I/usr/local/include"]
        if self.test_supports_compile_arg(arg_omp1, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp1)
                e.extra_link_args.append(arg_omp1)
        elif (sys.platform[:3].lower() == "dar") and self.test_supports_compile_arg(args_apple_omp, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args += ["-Xclang", "-fopenmp"]
                e.extra_link_args += ["-lomp"]
        elif (sys.platform[:3].lower() == "dar") and self.test_supports_compile_arg(args_apple_omp2, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args += ["-Xclang", "-fopenmp"]
                e.extra_link_args += ["-L/usr/local/lib", "-lomp"]
                e.include_dirs += ["/usr/local/include"]
        elif self.test_supports_compile_arg(arg_omp2, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp2)
                e.extra_link_args.append(arg_omp2)
        elif self.test_supports_compile_arg(arg_omp3, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp3)
                e.extra_link_args.append(arg_omp3)
        elif self.test_supports_compile_arg(arg_omp4, with_omp=True):
            for e in self.extensions:
                e.extra_compile_args.append(arg_omp4)
                e.extra_link_args.append(arg_omp4)
        else:
            set_omp_false()

    def test_supports_compile_arg(self, comm, with_omp=False):
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
                if not isinstance(self.compiler.compiler_cxx, list):
                    cmd = list(self.compiler.compiler_cxx)
                else:
                    cmd = self.compiler.compiler_cxx
            except:
                cmd = self.compiler.compiler_cxx
            val_good = subprocess.call(cmd + [fname])
            if with_omp:
                with open(fname, "w") as ftest:
                    ftest.write(u"#include <omp.h>\nint main(int argc, char**argv) {return 0;}\n")
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
                if not isinstance(self.compiler.compiler_cxx, list):
                    cmd = list(self.compiler.compiler_cxx)
                else:
                    cmd = self.compiler.compiler_cxx
            except:
                cmd = self.compiler.compiler_cxx
            val_good = subprocess.call(cmd + [fname])
            try:
                with open(fname, "w") as ftest:
                    ftest.write(u"int main(int argc, char**argv) {double *__restrict x = 0; return 0;}\n")
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
    version = '1.8.1-2',
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

"""SCons.Tool.nvcc

Tool-specific initialization for NVIDIA CUDA Compiler.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""

import SCons.Tool
import SCons.Scanner.C
import SCons.Defaults
import os
import platform


def get_cuda_paths():
    """Determines CUDA {bin,lib,include} paths

    returns (bin_path,lib_path,inc_path)
    """

    # determine defaults
    # if os.name == 'nt':
    #     bin_path = 'C:/CUDA/bin'
    #     lib_path = 'C:/CUDA/lib'
    #     inc_path = 'C:/CUDA/include'
    if os.name == 'posix':
        bin_path = '/public/software/compiler/rocm/dtk-24.04/bin'
        lib_path = '/public/software/compiler/rocm/dtk-24.04/lib'
        inc_path = '/public/software/compiler/rocm/dtk-24.04/include'
    else:
        raise ValueError('Error: unknown OS.  Where is nvcc installed?')

    lib_ext = ''
    if platform.machine()[-2:] == '64' and platform.platform()[:6] != 'Darwin':
        lib_ext = '64'

    # override with environment variables
    if 'DTK_ROOT' in os.environ:
        bin_path = os.path.join(os.path.abspath(os.environ['DTK_ROOT']), 'bin')
        lib_path = os.path.join(os.path.abspath(os.environ['DTK_ROOT']), 'lib')
        inc_path = os.path.join(os.path.abspath(os.environ['DTK_ROOT']), 'include')
    if 'HIP_BIN_PATH' in os.environ:
        bin_path = os.path.abspath(os.environ['HIP_BIN_PATH'])
    if 'HIP_LIB_PATH' in os.environ:
        lib_path = os.path.abspath(os.environ['HIP_LIB_PATH'])
    if 'HIP_INC_PATH' in os.environ:
        inc_path = os.path.abspath(os.environ['HIP_INC_PATH'])

    return (bin_path, lib_path + lib_ext, inc_path)


CUDASuffixes = ['.cu']

# make a CUDAScanner for finding #includes
# cuda uses the c preprocessor, so we can use the CScanner
CUDAScanner = SCons.Scanner.C.CScanner()


def add_common_nvcc_variables(env):
    """
    Add underlying common "NVIDIA CUDA compiler" variables that
    are used by multiple builders.
    """

    # "HIPCC common command line"
    if '_HIPCCOMCOM' not in env:
        # hipcc needs '-I' prepended before each include path, regardless of
        # platform
        env['_HIPCCWRAPCPPPATH'] = '${_concat("-I ", CPPPATH, "", __env__)}'
        # prepend -Xcompiler before each flag
        env['_HIPCCWRAPCFLAGS'] = '${_concat("-Xcompiler ", CFLAGS,     "", __env__)}'
        env['_HIPCCWRAPSHCFLAGS'] = '${_concat("-Xcompiler ", SHCFLAGS,   "", __env__)}'
        env['_HIPCCWRAPCCFLAGS'] = '${_concat("-Xcompiler ", CCFLAGS,   "", __env__)}'
        env['_HIPCCWRAPSHCCFLAGS'] = '${_concat("-Xcompiler ", SHCCFLAGS, "", __env__)}'
        # assemble the common command line
        env['_HIPCCOMCOM'] = '${_concat("-Xcompiler ", CPPFLAGS, "", __env__)} $_CPPDEFFLAGS $_HIPCCWRAPCPPPATH'


def generate(env):
    """
    Add Builders and construction variables for CUDA compilers to an Environment.
    """

    # create a builder that makes PTX files from .cu files
    ptx_builder = SCons.Builder.Builder(
        action='$HIPCC -ptx $HIPCCFLAGS $_HIPCCWRAPCFLAGS $HIPCCWRAPCCFLAGS $_HIPCCOMCOM $SOURCES -o $TARGET',
        emitter={},
        suffix='.ptx',
        src_suffix=CUDASuffixes)
    env['BUILDERS']['PTXFile'] = ptx_builder

    # create builders that make static & shared objects from .cu files
    static_obj, shared_obj = SCons.Tool.createObjBuilders(env)

    for suffix in CUDASuffixes:
        # Add this suffix to the list of things buildable by Object
        static_obj.add_action('$CUDAFILESUFFIX', '$HIPCCOM')
        shared_obj.add_action('$CUDAFILESUFFIX', '$SHHIPCCOM')
        static_obj.add_emitter(suffix, SCons.Defaults.StaticObjectEmitter)
        shared_obj.add_emitter(suffix, SCons.Defaults.SharedObjectEmitter)

        # Add this suffix to the list of things scannable
        SCons.Tool.SourceFileScanner.add_scanner(suffix, CUDAScanner)

    add_common_nvcc_variables(env)

    # set the "CUDA Compiler Command" environment variable
    # windows is picky about getting the full filename of the executable
    if os.name == 'nt':
        env['HIPCC'] = 'hipcc.exe'
        env['SHHIPCC'] = 'hipcc.exe'
    else:
        env['HIPCC'] = 'hipcc'
        env['SHHIPCC'] = 'hipcc'

    # set the include path, and pass both c compiler flags and c++ compiler
    # flags
    env['HIPCFLAGS'] = SCons.Util.CLVar('')
    env['SHHIPCFLAGS'] = SCons.Util.CLVar('') + ' -shared'

    # 'HIPCC Command'
    env['HIPCCOM'] = '$HIPCC -o $TARGET -c $HIPCFLAGS $_HIPCCWRAPCFLAGS $HIPCCWRAPCCFLAGS $_HIPCCOMCOM $SOURCES'
    env['SHHIPCCOM'] = '$SHHIPCC -o $TARGET -c $SHHIPCCFLAGS $_HIPCCWRAPSHCFLAGS $_HIPCCWRAPSHCCFLAGS $_HIPCCOMCOM $SOURCES'

    # the suffix of CUDA source files is '.cu'
    env['CUDAFILESUFFIX'] = '.cu'

    # XXX add code to generate builders for other miscellaneous
    # CUDA files here, such as .gpu, etc.

    # XXX intelligently detect location of nvcc and cuda libraries here
    (bin_path, lib_path, inc_path) = get_cuda_paths()

    env.PrependENVPath('PATH', bin_path)


def exists(env):
    return env.Detect('hipcc')

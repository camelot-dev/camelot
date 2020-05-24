#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ghostscript._gsprint - A low-level interface to the Ghostscript C-API using ctypes
"""
#
# Modifications 2018 by Vinayak Mehta <vmehta94@gmail.com>
# Copyright 2010-2018 by Hartmut Goebel <h.goebel@crazy-compilers.com>
#
# Display_callback Structure by Lasse Fister <commander@graphicore.de> in 2013
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import sys
from ctypes import *


# base/gserrors.h
#
# Internal code for a normal exit when usage info is displayed.
# This allows Window versions of Ghostscript to pause until
# the message can be read.
#
e_Info = -110

#
# Internal code for the .quit operator.
# The real quit code is an integer on the operand stack.
# gs_interpret returns this only for a .quit with a zero exit code.
#
e_Quit = -101

__author__ = "Hartmut Goebel <h.goebel@crazy-compilers.com>"
__copyright__ = "Copyright 2010-2018 by Hartmut Goebel <h.goebel@crazy-compilers.com>"
__license__ = "GNU General Public License version 3 (GPL v3)"
__version__ = "0.6"

gs_main_instance = c_void_p
display_callback = c_void_p

# https://www.ghostscript.com/doc/current/API.htm


class GhostscriptError(Exception):
    def __init__(self, ecode):
        self.code = ecode


def new_instance():
    """Create a new instance of Ghostscript

    This instance is passed to most other API functions.
    """
    # :todo: The caller_handle will be provided to callback functions.
    display_callback = None
    instance = gs_main_instance()
    rc = libgs.gsapi_new_instance(pointer(instance), display_callback)
    if rc != 0:
        raise GhostscriptError(rc)
    return instance


def delete_instance(instance):
    """Destroy an instance of Ghostscript

    Before you call this, Ghostscript must have finished.
    If Ghostscript has been initialised, you must call exit()
    before delete_instance()
    """
    return libgs.gsapi_delete_instance(instance)


if sys.platform == "win32":
    c_stdstream_call_t = WINFUNCTYPE(c_int, gs_main_instance, POINTER(c_char), c_int)
else:
    c_stdstream_call_t = CFUNCTYPE(c_int, gs_main_instance, POINTER(c_char), c_int)


def _wrap_stdin(infp):
    """Wrap a filehandle into a C function to be used as `stdin` callback
    for ``set_stdio``. The filehandle has to support the readline() method.
    """

    def _wrap(instance, dest, count):
        try:
            data = infp.readline(count)
        except:
            count = -1
        else:
            if not data:
                count = 0
            else:
                count = len(data)
                memmove(dest, c_char_p(data), count)
        return count

    return c_stdstream_call_t(_wrap)


def _wrap_stdout(outfp):
    """Wrap a filehandle into a C function to be used as `stdout` or
    `stderr` callback for ``set_stdio``. The filehandle has to support the
    write() and flush() methods.
    """

    def _wrap(instance, str, count):
        outfp.write(str[:count])
        outfp.flush()
        return count

    return c_stdstream_call_t(_wrap)


_wrap_stderr = _wrap_stdout


def set_stdio(instance, stdin, stdout, stderr):
    """Set the callback functions for stdio.

    ``stdin``, ``stdout`` and ``stderr`` have to be ``ctypes``
    callback functions matching the ``_gsprint.c_stdstream_call_t``
    prototype. You may want to use _wrap_* to wrap file handles.

    Note 1: This function only changes stdio of the Postscript
    interpreter, not that of the devices.

    Note 2: Make sure you keep references to C function objects
    as long as they are used from C code. Otherwise they may be
    garbage collected, crashing your program when a callback is made.

    The ``stdin`` callback function should return the number of
    characters read, `0` for EOF, or `-1` for error. The `stdout` and
    `stderr` callback functions should return the number of characters
    written.

    You may pass ``None`` for any of stdin, stdout or stderr , in which
    case the system stdin, stdout resp. stderr will be used.
    """
    rc = libgs.gsapi_set_stdio(instance, stdin, stdout, stderr)
    if rc not in (0, e_Quit, e_Info):
        raise GhostscriptError(rc)
    return rc


def init_with_args(instance, argv):
    """Initialise the interpreter

    1. If quit or EOF occur during init_with_args(), the return value
       will be e_Quit. This is not an error. You must call exit() and
       must not call any other functions.

    2. If usage info should be displayed, the return value will be
       e_Info which is not an error. Do not call exit().

    3. Under normal conditions this returns 0. You would then call one
       or more run_*() functions and then finish with exit()
    """
    ArgArray = c_char_p * len(argv)
    c_argv = ArgArray(*argv)
    rc = libgs.gsapi_init_with_args(instance, len(argv), c_argv)
    if rc not in (0, e_Quit, e_Info):
        raise GhostscriptError(rc)
    return rc


def exit(instance):
    """Exit the interpreter

    This must be called on shutdown if init_with_args() has been
    called, and just before delete_instance()
    """
    rc = libgs.gsapi_exit(instance)
    if rc != 0:
        raise GhostscriptError(rc)
    return rc


def __win32_finddll():
    try:
        import winreg
    except ImportError:
        # assume Python 2
        from _winreg import (
            OpenKey,
            CloseKey,
            EnumKey,
            QueryValueEx,
            QueryInfoKey,
            HKEY_LOCAL_MACHINE,
        )
    else:
        from winreg import (
            OpenKey,
            CloseKey,
            EnumKey,
            QueryValueEx,
            QueryInfoKey,
            HKEY_LOCAL_MACHINE,
        )

    from distutils.version import LooseVersion
    import os

    dlls = []
    # Look up different variants of Ghostscript and take the highest
    # version for which the DLL is to be found in the filesystem.
    for key_name in (
        "AFPL Ghostscript",
        "Aladdin Ghostscript",
        "GNU Ghostscript",
        "GPL Ghostscript",
    ):
        try:
            k1 = OpenKey(HKEY_LOCAL_MACHINE, "Software\\%s" % key_name)
            for num in range(0, QueryInfoKey(k1)[0]):
                version = EnumKey(k1, num)
                try:
                    k2 = OpenKey(k1, version)
                    dll_path = QueryValueEx(k2, "GS_DLL")[0]
                    CloseKey(k2)
                    if os.path.exists(dll_path):
                        dlls.append((LooseVersion(version), dll_path))
                except WindowsError:
                    pass
            CloseKey(k1)
        except WindowsError:
            pass
    if dlls:
        dlls.sort()
        return dlls[-1][-1]
    else:
        return None


if sys.platform == "win32":
    libgs = __win32_finddll()
    if not libgs:
        import ctypes.util

        libgs = ctypes.util.find_library(
            "".join(("gsdll", str(ctypes.sizeof(ctypes.c_voidp) * 8), ".dll"))
        )  # finds in %PATH%
    if not libgs:
        raise RuntimeError("Please make sure that Ghostscript is installed")
    libgs = windll.LoadLibrary(libgs)
else:
    try:
        libgs = cdll.LoadLibrary("libgs.so")
    except OSError:
        # shared object file not found
        import ctypes.util

        libgs = ctypes.util.find_library("gs")
        if not libgs:
            raise RuntimeError("Please make sure that Ghostscript is installed")
        libgs = cdll.LoadLibrary(libgs)

del __win32_finddll

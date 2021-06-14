#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ghostscript - A Python interface for the Ghostscript interpreter C-API
"""
#
# Modifications 2018 by Vinayak Mehta <vmehta94@gmail.com>
# Copyright 2010-2018 by Hartmut Goebel <h.goebel@crazy-compilers.com>
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

from . import _gsprint as gs


__author__ = "Hartmut Goebel <h.goebel@crazy-compilers.com>"
__copyright__ = "Copyright 2010-2018 by Hartmut Goebel <h.goebel@crazy-compilers.com>"
__license__ = "GNU General Public License version 3 (GPL v3)"
__version__ = "0.6"


class __Ghostscript(object):
    def __init__(self, instance, args, stdin=None, stdout=None, stderr=None):
        self._initialized = False
        self._callbacks = None
        if stdin or stdout or stderr:
            self.set_stdio(stdin, stdout, stderr)
        rc = gs.init_with_args(instance, args)
        self._initialized = True
        if rc == gs.e_Quit:
            self.exit()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.exit()

    def set_stdio(self, stdin=None, stdout=None, stderr=None):
        """Set stdin, stdout and stderr of the ghostscript interpreter.

        The ``stdin`` stream has to support the ``readline()``
        interface. The ``stdout`` and ``stderr`` streams have to
        support the ``write()`` and ``flush()`` interface.

        Please note that this does not affect the input- and output-
        streams of the devices. Esp. setting stdout does not allow
        catching the devise-output even when using ``-sOutputFile=-``.

        """
        global __instance__
        self._callbacks = (
            stdin and gs._wrap_stdin(stdin) or None,
            stdout and gs._wrap_stdout(stdout) or None,
            stderr and gs._wrap_stderr(stderr) or None,
        )
        gs.set_stdio(__instance__, *self._callbacks)

    def __del__(self):
        self.exit()

    def exit(self):
        global __instance__
        if self._initialized:
            if __instance__ is not None:
                gs.exit(__instance__)
                gs.delete_instance(__instance__)
                __instance__ = None
            self._initialized = False


def Ghostscript(*args, **kwargs):
    """Factory function for setting up a Ghostscript instance"""
    global __instance__
    # Ghostscript only supports a single instance
    if __instance__ is None:
        __instance__ = gs.new_instance()
    return __Ghostscript(
        __instance__,
        args,
        stdin=kwargs.get("stdin", None),
        stdout=kwargs.get("stdout", None),
        stderr=kwargs.get("stderr", None),
    )


__instance__ = None

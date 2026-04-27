"""
cffi loader for the BiRewire C backend.

Exports:
  _C_AVAILABLE : bool   — True when the compiled shared library loaded OK
  ffi          : FFI    — the cffi FFI object (or None)
  lib          : CLib   — the loaded library (or None)
"""

try:
    from pybirewirex._birewire import ffi, lib  # type: ignore[import]

    _C_AVAILABLE: bool = True
except ImportError:
    ffi = None  # type: ignore[assignment]
    lib = None  # type: ignore[assignment]
    _C_AVAILABLE = False

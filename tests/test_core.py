"""Tests for Issue #2: package scaffold and C core loader."""

import importlib
import os
import sys

import pytest


def test_import_pybirewirex():
    import pybirewirex

    assert hasattr(pybirewirex, "_C_AVAILABLE")


def test_c_available_true():
    from pybirewirex._core import _C_AVAILABLE, ffi, lib

    assert _C_AVAILABLE is True
    assert ffi is not None
    assert lib is not None


@pytest.mark.skipif(
    # Once dlopen() loads a shared library it stays resident in the process even
    # after the file is renamed, so the reload trick cannot evict the C extension.
    # This test is only meaningful in environments built without the C backend.
    __import__("pybirewirex")._C_AVAILABLE,
    reason="C extension already resident in process; cannot be unloaded to test fallback path",
)
def test_c_available_false_when_so_absent(tmp_path):
    """Simulate missing .so by temporarily renaming it."""
    so = _find_so()
    if so is None:
        pytest.skip("cannot locate .so for this test")

    backup = str(so) + ".bak"
    os.rename(so, backup)
    try:
        _reload_core()
        from pybirewirex._core import _C_AVAILABLE

        assert _C_AVAILABLE is False
    finally:
        os.rename(backup, so)
        _reload_core()  # restore for subsequent tests


def test_lib_exposes_rewire_bipartite():
    from pybirewirex._core import lib

    assert callable(lib.bw_rewire_bipartite)


def test_lib_exposes_analysis_bipartite():
    from pybirewirex._core import lib

    assert callable(lib.bw_analysis_bipartite)


def test_lib_exposes_rewire_undirected():
    from pybirewirex._core import lib

    assert callable(lib.bw_rewire_undirected)


def test_lib_exposes_analysis_undirected():
    from pybirewirex._core import lib

    assert callable(lib.bw_analysis_undirected)


def test_lib_exposes_rewire_sparse_bipartite():
    from pybirewirex._core import lib

    assert callable(lib.bw_rewire_sparse_bipartite)


def test_lib_exposes_rewire_sparse():
    from pybirewirex._core import lib

    assert callable(lib.bw_rewire_sparse)


def _find_so() -> str | None:
    import glob

    candidates = glob.glob(
        os.path.join(os.path.dirname(__file__), "..", "pybirewirex", "_birewire*.so")
    )
    return candidates[0] if candidates else None


def _reload_core():
    to_remove = [k for k in sys.modules if "birewire" in k]
    for k in to_remove:
        del sys.modules[k]
    importlib.import_module("pybirewirex._core")

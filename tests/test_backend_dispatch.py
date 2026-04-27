"""Backend registry dispatch behavior."""

import pytest

from fairlearn_fhe._backends import get_backend, get_default_backend, set_default_backend


def test_default_backend_roundtrip():
    original = get_default_backend()
    try:
        set_default_backend("tenseal")
        assert get_default_backend() == "tenseal"
        assert get_backend().__name__.endswith("tenseal_backend")
        assert get_backend("openfhe").__name__.endswith("openfhe_backend")
    finally:
        set_default_backend(original)


def test_backend_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown backend"):
        set_default_backend("bad")
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("bad")

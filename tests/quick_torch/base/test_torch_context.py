import pytest

from tests.quick.base.test_context import _test_to_dict

pytest.importorskip("torch")


def test_to_dict():
    _test_to_dict("torch")

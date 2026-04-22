import pytest

from tests.quick.base.test_context import _test_to_dict

pytest.importorskip("tensorflow")


def test_to_dict():
    _test_to_dict("tensorflow")

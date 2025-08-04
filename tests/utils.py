import inspect
from typing import Any

import numpy as np
import pytest


def assert_equal(actual: Any, expected: Any) -> None:
    """型を問わず比較するための共通assert関数"""
    if expected is None:
        assert actual is None

    if isinstance(expected, (int, bool, str, float)):
        assert actual == pytest.approx(expected)
    elif isinstance(expected, (list, tuple)):
        assert type(actual) == type(expected), f"type mismatch: {type(actual)} != {type(expected)}"
        assert len(actual) == len(expected), f"length mismatch: {len(actual)} != {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            try:
                assert_equal(a, e)
            except AssertionError as e2:
                raise AssertionError(f"index {i}: {e2}") from e2
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), f"dict keys mismatch: {actual.keys()} != {expected.keys()}"
        for k in expected:
            try:
                assert_equal(actual[k], expected[k])
            except AssertionError as e2:
                raise AssertionError(f"key {k}: {e2}") from e2
    elif isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray), f"type mismatch: {type(actual)} != {type(expected)}"
        assert actual.shape == expected.shape, f"shape mismatch: {actual.shape} != {expected.shape}"
        if np.issubdtype(expected.dtype, np.floating):
            assert np.allclose(actual, expected), f"array values mismatch: {actual} != {expected}"
        else:
            assert np.array_equal(actual, expected), f"array values mismatch: {actual} != {expected}"
    elif callable(expected):  # objectの前
        assert actual is expected, f"value mismatch: {actual} != {expected}"
    elif inspect.isclass(type(expected)) and type(expected).__module__ != "builtins":
        if "__eq__" in type(expected).__dict__:
            assert actual == expected, f"value mismatch: {actual} != {expected}"
        else:
            assert type(actual) == type(expected), f"type mismatch: {type(actual)} != {type(expected)}"
    else:
        assert actual == expected, f"value mismatch: {actual} != {expected}"

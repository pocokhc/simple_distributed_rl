import enum
import json
from dataclasses import dataclass
from pprint import pprint

import numpy as np

from srl.utils import serialize


class DummyEnum(enum.Enum):
    a = enum.auto
    b = enum.auto
    c = enum.auto


@dataclass
class DummyDataClassClass:
    val: float = 1.1


class DummyClass:
    def __init__(self):
        self.a = 1


def dummy_func(a):
    return a


def test_convert_for_json():
    d = {
        "none": None,
        "int": 1,
        "float": 1.2,
        "bool": True,
        "str": "AAA",
        "bytes": b"BBB",
        "list": [1, "a"],
        "tuple": (2.2, "cc"),
        "dict": {"a": 2, "b": "b", "c": [1, 2, 3]},
        "enum": DummyEnum.a,
        "dataclass": DummyDataClassClass(),
        "np1": np.array((1, 2, 3)),
        "np2": np.zeros(()),
        "class": DummyClass(),
        "func": dummy_func,
    }
    d2 = serialize.convert_for_json(d)
    pprint(d2)
    json.dumps(d2)


def test_serialize_for_json():
    d = {
        "none": None,
        "int": 1,
        "float": 1.2,
        "bool": True,
        "str": "AAA",
        "bytes": b"BBB",
        "list": [1, "a"],
        "tuple": (2.2, "cc"),
        "dict": {"a": 2, "b": "b", "c": [1, 2, 3]},
        "enum": DummyEnum.a,
        "dataclass": DummyDataClassClass(),
        "np1": np.array((1, 2, 3)),
        "np2": np.zeros(()),
        "class": DummyClass(),
        "func": dummy_func,
    }
    d2, base_type = serialize.serialize_for_json(d)
    pprint(d2)
    pprint(base_type)
    json.dumps(d2)

    d3 = serialize.deserialize_for_json(d2, base_type)
    pprint(d3)

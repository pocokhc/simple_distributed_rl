import enum
import json
from dataclasses import dataclass, field
from pprint import pprint
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytest

from srl.utils import serialize
from tests.utils import assert_equal


class DummyEnum(enum.Enum):
    a = enum.auto()
    b = enum.auto()
    c = enum.auto()


@dataclass
class DummyDataClass:
    val: float = 1.1


class DummyClass:
    def __init__(self):
        self.a = 1


class DummyClassDict:
    def __init__(self, a=1.1):
        self.a = a

    def from_dict(self, dat: dict):
        self.a = dat["a"]

    def to_dict(self) -> dict:
        return {"a": self.a}


def dummy_func(a):
    return a


def test_dataclass_to_print():
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
        "dataclass": DummyDataClass(),
        "np1": np.array((1, 2, 3)),
        "np2": np.zeros(()),
        "class": DummyClass(),
        "func": dummy_func,
    }
    d2 = serialize.dataclass_to_dict(d, to_print=True)
    pprint(d2)
    json.dumps(d2)


@dataclass
class DummyTestClass:
    s_none: None = None
    s_int: int = 0
    o_int: Optional[int] = None
    s_float: float = 0.0
    o_float: Optional[float] = None
    s_bool: bool = False
    o_bool: Optional[bool] = None
    s_str: str = ""
    o_str: Optional[str] = None
    s_bytes: bytes = b""
    s_bytes2: bytes = b""
    o_bytes: Optional[bytes] = None
    s_list: list = field(default_factory=list)
    s_list2: List[Union[int, str]] = field(default_factory=list)
    o_list: Optional[list] = None
    s_tuple: Tuple[float, str] = field(default_factory=tuple)
    o_tuple: Optional[Tuple[float, str]] = None
    s_dict: dict = field(default_factory=dict)
    s_dict2: Dict[str, Union[int, str, List[int]]] = field(default_factory=dict)
    o_dict: Optional[dict] = None
    s_enum: DummyEnum = DummyEnum.a
    s_enum2: DummyEnum = DummyEnum.a
    o_enum: Optional[DummyEnum] = None
    s_dataclass: DummyDataClass = field(default_factory=lambda: DummyDataClass())
    s_dataclass2: DummyDataClass = field(default_factory=lambda: DummyDataClass())
    o_dataclass: Optional[DummyDataClass] = None
    s_np: np.ndarray = field(default_factory=lambda: np.zeros(1))
    s_np2: np.ndarray = field(default_factory=lambda: np.zeros(1))
    o_np: Optional[np.ndarray] = None
    s_dict_class: DummyClassDict = field(default_factory=lambda: DummyClassDict())
    o_dict_class: Optional[DummyClassDict] = None


@pytest.mark.parametrize("strict", [False, True])
def test_dataclass(tmpdir, strict):
    d = {
        "s_none": None,
        "s_int": 3,
        "o_int": 3,
        "s_float": 1.2,
        "o_float": 1.2,
        "s_bool": True,
        "o_bool": True,
        "s_str": "AAA",
        "o_str": "AAA",
        "s_bytes": b"BBB",
        "s_bytes2": "QkJC",
        "o_bytes": b"BBB",
        "s_list": [1, "a"],
        "s_list2": [1, "a"],
        "o_list": [1, "a"],
        "s_tuple": (2.2, "cc"),
        "o_tuple": (2.2, "cc"),
        "s_dict": {"a": 2, "b": "b", "c": [1, 2, 3]},
        "s_dict2": {"a": 2, "b": "b", "c": [1, 2, 3]},
        "o_dict": {"a": 2, "b": "b", "c": [1, 2, 3]},
        "s_enum": DummyEnum.b,
        "s_enum2": "b",
        "o_enum": DummyEnum.b,
        "s_dataclass": DummyDataClass(2.2),
        "s_dataclass2": {"val": 2.2},
        "o_dataclass": {"val": 2.2},
        "s_np": np.array((1, 2, 3)),
        "s_np2": [1, 2, 3],
        "o_np": np.array((1, 2, 3)),
        "s_dict_class": DummyClassDict(2.2),
        "o_dict_class": DummyClassDict(2.2),
    }
    pprint(d)
    o = DummyTestClass()
    serialize.apply_dict_to_dataclass(o, d, strict)
    pprint(o)
    assert_equal(o.s_none, None)
    assert_equal(o.s_int, 3)
    assert_equal(o.o_int, 3)
    assert_equal(o.s_float, 1.2)
    assert_equal(o.o_float, 1.2)
    assert_equal(o.s_bool, True)
    assert_equal(o.o_bool, True)
    assert_equal(o.s_str, "AAA")
    assert_equal(o.o_str, "AAA")
    assert_equal(o.s_bytes, b"BBB")
    assert_equal(o.s_bytes2, b"BBB")
    assert_equal(o.o_bytes, b"BBB")
    assert_equal(o.s_list, [1, "a"])
    assert_equal(o.s_list2, [1, "a"])
    assert_equal(o.o_list, [1, "a"])
    assert_equal(o.s_tuple, (2.2, "cc"))
    assert_equal(o.o_tuple, (2.2, "cc"))
    assert_equal(o.s_dict, {"a": 2, "b": "b", "c": [1, 2, 3]})
    assert_equal(o.s_dict2, {"a": 2, "b": "b", "c": [1, 2, 3]})
    assert_equal(o.o_dict, {"a": 2, "b": "b", "c": [1, 2, 3]})
    assert_equal(o.s_enum, DummyEnum.b)
    assert_equal(o.s_enum2, DummyEnum.b)
    assert_equal(o.o_enum, DummyEnum.b)
    assert_equal(o.s_dataclass.val, 2.2)
    assert_equal(o.s_dataclass2.val, 2.2)
    assert o.o_dataclass is not None
    assert_equal(o.o_dataclass.val, 2.2)
    assert_equal(o.s_np, np.array((1, 2, 3)))
    assert_equal(o.s_np2, np.array((1, 2, 3)))
    assert_equal(o.o_np, np.array((1, 2, 3)))
    assert_equal(o.s_dict_class.a, 2.2)
    assert o.o_dict_class is not None
    assert_equal(o.o_dict_class.a, 2.2)

    d["s_bytes"] = "QkJC"
    d["o_bytes"] = "QkJC"
    d["s_enum"] = "b"
    d["o_enum"] = "b"
    d["s_dataclass"] = {"_target_": "utils.test_serialize.DummyDataClass", "val": 2.2}
    d["s_dataclass2"] = {"_target_": "utils.test_serialize.DummyDataClass", "val": 2.2}
    d["o_dataclass"] = {"_target_": "utils.test_serialize.DummyDataClass", "val": 2.2}
    d["s_np"] = [1, 2, 3]
    d["o_np"] = [1, 2, 3]
    d["s_dict_class"] = {"_target_": "utils.test_serialize.DummyClassDict", "a": 2.2}
    d["o_dict_class"] = {"_target_": "utils.test_serialize.DummyClassDict", "a": 2.2}
    d2 = serialize.dataclass_to_dict(o)
    for k, v in d2.items():
        if k in {"_target_", "_args_"}:
            continue
        print(k, d[k])
        assert_equal(d[k], v)

    # --- ファイル経由でも変わらない事
    path = tmpdir / "tmp.dat"
    serialize.save_dict(serialize.dataclass_to_dict(o), path)
    o = serialize.apply_dict_to_dataclass(None, serialize.load_dict(str(path)), strict)
    pprint(o)
    assert_equal(o.s_none, None)
    assert_equal(o.s_int, 3)
    assert_equal(o.o_int, 3)
    assert_equal(o.s_float, 1.2)
    assert_equal(o.o_float, 1.2)
    assert_equal(o.s_bool, True)
    assert_equal(o.o_bool, True)
    assert_equal(o.s_str, "AAA")
    assert_equal(o.o_str, "AAA")
    assert_equal(o.s_bytes, b"BBB")
    assert_equal(o.s_bytes2, b"BBB")
    assert_equal(o.o_bytes, b"BBB")
    assert_equal(o.s_list, [1, "a"])
    assert_equal(o.s_list2, [1, "a"])
    assert_equal(o.o_list, [1, "a"])
    assert_equal(o.s_tuple, (2.2, "cc"))
    assert_equal(o.o_tuple, (2.2, "cc"))
    assert_equal(o.s_dict, {"a": 2, "b": "b", "c": [1, 2, 3]})
    assert_equal(o.s_dict2, {"a": 2, "b": "b", "c": [1, 2, 3]})
    assert_equal(o.o_dict, {"a": 2, "b": "b", "c": [1, 2, 3]})
    assert_equal(o.s_enum, DummyEnum.b)
    assert_equal(o.s_enum2, DummyEnum.b)
    assert_equal(o.o_enum, DummyEnum.b)
    assert_equal(o.s_dataclass.val, 2.2)
    assert_equal(o.s_dataclass2.val, 2.2)
    assert o.o_dataclass is not None
    assert_equal(o.o_dataclass.val, 2.2)
    assert_equal(o.s_np, np.array((1, 2, 3)))
    assert_equal(o.s_np2, np.array((1, 2, 3)))
    assert_equal(o.o_np, np.array((1, 2, 3)))
    assert_equal(o.s_dict_class.a, 2.2)
    assert o.o_dict_class is not None
    assert_equal(o.o_dict_class.a, 2.2)


class Dummy2:
    def __init__(self, param1=0, param2=0, b=0) -> None:
        self.param1 = param1
        self.param2 = param2
        self.b = b


class Dummy1:
    def __init__(self, param1, param2) -> None:
        self.param1 = param1
        self.param2 = param2


def test_dataclass_instance1():
    cfg = {
        "_target_": __name__ + ".Dummy1",
        "_args_": [
            {
                "_target_": __name__ + ".Dummy2",
                "_args_": ["a", "a"],
            },
            {
                "_target_": __name__ + ".Dummy2",
                "_args_": ["b", "b"],
                "b": 1,
            },
        ],
    }
    o: Dummy1 = serialize.apply_dict_to_dataclass(None, cfg)
    assert isinstance(o, Dummy1)
    assert isinstance(o.param1, Dummy2)
    assert o.param1.param1 == "a"
    assert o.param1.param2 == "a"
    assert isinstance(o.param2, Dummy2)
    assert o.param2.param1 == "b"
    assert o.param2.param2 == "b"
    assert o.param2.b == 1


def test_dataclass_instance2():
    cfg = {
        "_target_": __name__ + ".Dummy1",
        "param1": {
            "_target_": __name__ + ".Dummy2",
            "param1": "a",
            "param2": "a",
        },
        "param2": {
            "_target_": __name__ + ".Dummy2",
            "_args_": ["b", "b"],
            "b": 1,
        },
    }
    o = Dummy1(Dummy2("", "", 5), None)
    o: Dummy1 = serialize.apply_dict_to_dataclass(o, cfg)
    assert isinstance(o, Dummy1)
    assert isinstance(o.param1, Dummy2)
    assert o.param1.param1 == "a"
    assert o.param1.param2 == "a"
    assert o.param1.b == 5
    assert isinstance(o.param2, Dummy2)
    assert o.param2.param1 == "b"
    assert o.param2.param2 == "b"
    assert o.param2.b == 1

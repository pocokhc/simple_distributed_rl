import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


class Info:
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._data_type: Dict[str, str] = {}

    def __setitem__(self, key, value):
        self.set_scalar(key, value)

    def __getitem__(self, key):
        return self._data[key]

    def __delitem__(self, key):
        del self._data[key]

    def __str__(self) -> str:
        return str(self._data)

    def to_dict(self):
        return self._data

    def items(self):
        return self._data.items()

    def copy(self):
        o = Info()
        o._data = self._data.copy()
        o._data_type = self._data_type.copy()
        return o

    def set_scalar(
        self,
        name: str,
        data: Any,
        data_type: str = "",
        # aggregation_type: str = "ave",
    ):
        self._data[name] = data
        self._data_type[name] = data_type

    def set_dict(self, d: dict):
        for k, v in d.items():
            self.set_scalar(k, v)

    def update(self, d: dict):
        for k, v in d.items():
            self.set_scalar(k, v)

    # -------------------------
    def to_str(self) -> str:
        s = ""
        for k, data in self._data.items():
            d_type = self._data_type[k]
            if d_type == "":
                if isinstance(data, int):
                    d_type = "int"
                elif isinstance(data, float):
                    d_type = "float"
                elif isinstance(data, np.integer):
                    d_type = "int"
                elif isinstance(data, np.floating):
                    d_type = "float"
                elif isinstance(data, np.ndarray):
                    d_type = "float"

            # --- print str
            s += f"|{k} "
            if d_type == "int":
                s += f"{int(data):2d}"
            elif d_type == "float":
                data = float(data)
                if -10 <= data <= 10:
                    s += f"{data:.3f}"
                else:
                    s += f"{data:.1f}"
            else:
                s += f"{str(data)}"

        return s

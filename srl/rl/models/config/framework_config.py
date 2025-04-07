from dataclasses import dataclass
from typing import Literal

from srl.base.exception import UndefinedError
from srl.utils.common import is_package_installed


@dataclass
class RLConfigComponentFramework:
    framework: Literal["auto", "tensorflow", "torch"] = "auto"

    def set_tensorflow(self):
        """use tensorflow"""
        self.framework = "tensorflow"
        return self

    def set_torch(self):
        """use torch"""
        self.framework = "torch"
        return self

    def set_auto(self):
        """use tensorflow/torch
        インストールされている方を採用します。
        両方インストールされている場合はTensorflowが優先されます。
        """
        self.framework = "auto"
        return self

    def get_framework(self) -> str:
        if self.framework == "auto":
            if is_package_installed("tensorflow"):
                return "tensorflow"
            elif is_package_installed("torch"):
                return "torch"
            else:
                raise UndefinedError("'tensorflow' or 'torch' could not be found.")

        return self.framework

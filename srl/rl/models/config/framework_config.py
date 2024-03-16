from dataclasses import dataclass, field

from srl.utils.common import is_package_installed


@dataclass
class FrameworkConfig:
    _framework: str = field(init=False, default="")

    def set_tensorflow(self):
        """use tensorflow"""
        self._framework = "tensorflow"

    def set_torch(self):
        """use torch"""
        self._framework = "torch"

    def set_auto(self):
        """use tensorflow/torch
        インストールされている方を採用します。
        両方インストールされている場合はTensorflowが優先されます。
        """
        self._framework = ""

    # ---------------------

    def get_use_framework(self, enable_assert: bool = True) -> str:
        if self._framework != "":
            return self._framework

        if is_package_installed("tensorflow"):
            return "tensorflow"
        elif is_package_installed("torch"):
            return "torch"

        if enable_assert:
            assert False, "'tensorflow' or 'torch' could not be found."
        return ""

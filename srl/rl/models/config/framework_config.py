from dataclasses import dataclass, field

from srl.base.exception import UndefinedError
from srl.rl.models.config.image_block import ImageBlockConfig
from srl.rl.models.config.mlp_block import MLPBlockConfig
from srl.utils.common import is_package_installed


@dataclass
class RLConfigComponentFramework:
    framework: str = "auto"

    #: <:ref:`MLPBlock`> This layer is only used when the input is an value.
    input_value_block: MLPBlockConfig = field(
        init=False,
        default_factory=lambda: MLPBlockConfig().set(()),
    )

    #: <:ref:`ImageBlock`> This layer is only used when the input is an image.
    input_image_block: ImageBlockConfig = field(
        init=False,
        default_factory=lambda: ImageBlockConfig(),
    )

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

    def create_framework_str(self) -> str:
        if self.framework == "auto":
            if is_package_installed("tensorflow"):
                return "tensorflow"
            elif is_package_installed("torch"):
                return "torch"
            else:
                raise UndefinedError("'tensorflow' or 'torch' could not be found.")

        return self.framework

    def assert_params_framework(self):
        pass

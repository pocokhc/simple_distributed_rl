from srl.utils import common

from . import base_agent57, base_agent57_light, base_dqn, base_rainbow

common.logger_print()


class Test_agent57_CPU(base_agent57.BaseCase):
    def get_framework(self) -> str:
        return "torch"

    def get_device(self) -> str:
        return "CPU"


class Test_agent57_GPU(base_agent57.BaseCase):
    def get_framework(self) -> str:
        return "torch"

    def get_device(self) -> str:
        return "GPU"


class Test_agent57_light_CPU(base_agent57_light.BaseCase):
    def get_framework(self) -> str:
        return "torch"

    def get_device(self) -> str:
        return "CPU"


class Test_agent57_light_GPU(base_agent57_light.BaseCase):
    def get_framework(self) -> str:
        return "torch"

    def get_device(self) -> str:
        return "GPU"


class Test_dqn_CPU(base_dqn.BaseCase):
    def get_framework(self) -> str:
        return "torch"

    def get_device(self) -> str:
        return "CPU"


class Test_dqn_GPU(base_dqn.BaseCase):
    def get_framework(self) -> str:
        return "torch"

    def get_device(self) -> str:
        return "GPU"

    def test_image_r2d3(self):
        self.case_image_r2d3()


class Test_rainbow_CPU(base_rainbow.BaseCase):
    def get_framework(self) -> str:
        return "torch"

    def get_device(self) -> str:
        return "CPU"


class Test_rainbow_GPU(base_rainbow.BaseCase):
    def get_framework(self) -> str:
        return "torch"

    def get_device(self) -> str:
        return "GPU"

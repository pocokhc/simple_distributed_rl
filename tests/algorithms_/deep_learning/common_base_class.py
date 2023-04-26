from abc import ABC

from srl import runner
from srl.test import TestRL


class CommonBaseClass(ABC):
    def return_params(self):
        raise NotImplementedError()

    def return_rl_config(self, framework):
        raise NotImplementedError()

    def create_config(self, env):
        framework, device = self.return_params()
        rl_config = self.return_rl_config(framework)
        device_main = "CPU"
        device_mp_trainer = "CPU"
        device_mp_actors = "CPU"
        if device == "GPU":
            device_main = "AUTO"
            device_mp_trainer = "GPU"
            device_mp_actors = "CPU"
        config = runner.Config(
            env,
            rl_config,
            device_main=device_main,
            device_mp_trainer=device_mp_trainer,
            device_mp_actors=device_mp_actors,
            seed=1,
            seed_enable_gpu=True,
        )
        return config, rl_config, TestRL()

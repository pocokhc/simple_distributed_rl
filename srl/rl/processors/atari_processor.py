from srl.base.env.env_run import EnvRun
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import Processor
from srl.utils.common import is_package_installed


class AtariProcessor(Processor):
    def __init__(self):
        assert is_package_installed("ale_py")

    def setup(self, env: EnvRun, rl_config: RLConfig):
        self.enable = "AtariEnv" in str(env.unwrapped)

    def on_reset(self, env: EnvRun):
        if not self.enable:
            return
        self.lives = env.unwrapped.env.unwrapped.ale.lives()

    def preprocess_done(self, done: bool, env: EnvRun) -> bool:
        if not self.enable:
            return done
        new_lives = env.unwrapped.env.unwrapped.ale.lives()
        if new_lives < self.lives:
            return True
        self.lives = new_lives
        return done

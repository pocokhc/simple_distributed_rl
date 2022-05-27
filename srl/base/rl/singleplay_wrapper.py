from typing import Optional

from srl.base.define import EnvAction, Info
from srl.base.env.base import EnvRun
from srl.base.rl.base import RLWorker


class SinglePlayWorkerWrapper:
    def __init__(self, worker: RLWorker):
        self.worker = worker

    # ------------------------------------
    # episode properties
    # ------------------------------------
    @property
    def player_index(self) -> int:
        return self.worker.player_index

    @property
    def info(self) -> Optional[Info]:
        return self.worker.info

    # ------------------------------------
    # episode functions
    # ------------------------------------
    def on_reset(self, env: EnvRun) -> None:
        self.worker.on_reset(env, 0)

    def policy(self, env: EnvRun) -> EnvAction:
        return self.worker.policy(env)

    def on_step(self, env: EnvRun) -> Info:
        info = self.worker.on_step(env)
        assert info is not None
        return info

    def render(self, env: EnvRun) -> None:
        self.worker.render(env)

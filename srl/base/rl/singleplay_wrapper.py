from typing import Optional

from srl.base.define import EnvAction, Info
from srl.base.env.base import EnvRun
from srl.base.rl.base import WorkerRun


class SinglePlayWorkerWrapper:
    def __init__(self, worker: WorkerRun):
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
        self.action = self.worker.policy(env)

    def policy(self, env: EnvRun) -> EnvAction:
        return self.action

    def on_step(self, env: EnvRun) -> Info:
        self.worker.on_step(env)
        if not env.done:
            self.action = self.worker.policy(env)
        assert self.worker.info is not None
        return self.worker.info

    def render(self, env: EnvRun) -> None:
        self.worker.render(env)

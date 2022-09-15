from typing import Optional, Union

from srl.base.define import EnvAction, Info, PlayRenderMode
from srl.base.env.base import EnvRun
from srl.base.rl.worker import WorkerRun


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

    @property
    def reward(self) -> float:
        return self.worker.reward

    # ------------------------------------
    # episode functions
    # ------------------------------------
    def on_reset(self, env: EnvRun, mode: Union[str, PlayRenderMode] = "") -> None:
        self.worker.set_render_mode(mode)
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

    def set_render_mode(self, mode: Union[str, PlayRenderMode]) -> None:
        self.worker.set_render_mode(mode)

    def render(self, env: EnvRun) -> None:
        self.worker.render(env)

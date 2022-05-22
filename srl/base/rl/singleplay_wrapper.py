from srl.base.define import EnvAction, EnvObservation, Info
from srl.base.env.base import EnvBase
from srl.base.rl.base import RLWorker


class SinglePlayWorkerWrapper:
    def __init__(self, worker: RLWorker):
        self.worker = worker

    def on_reset(self, state: EnvObservation, env: EnvBase) -> None:
        self.worker.on_reset(state, 0, env)

    def policy(self, state: EnvObservation, env: EnvBase) -> EnvAction:
        return self.worker.policy(state, 0, env)

    def on_step(
        self,
        next_state: EnvObservation,
        reward: float,
        done: bool,
        env: EnvBase,
    ) -> Info:
        return self.worker.on_step(next_state, reward, done, 0, env)

    def render(self, env: EnvBase) -> None:
        self.worker.render(env, 0)

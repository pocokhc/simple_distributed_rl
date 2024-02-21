from typing import Tuple

import srl
from srl.base.define import InfoType, RLActionType
from srl.base.rl.base import DummyRLMemory, DummyRLParameter, RLWorker
from srl.base.rl.config import DummyRLConfig
from srl.base.rl.worker_run import WorkerRun
from srl.base.run.context import RunContext
from srl.base.run.core_play import play


class StubWorker(RLWorker):
    def on_reset(self, worker: WorkerRun) -> InfoType:
        inv_acts = [i for i, s in enumerate(worker.state) if s != 0]
        assert inv_acts == worker.get_invalid_actions()
        return {}

    def policy(self, worker: WorkerRun) -> Tuple[RLActionType, InfoType]:
        inv_acts = [i for i, s in enumerate(worker.state) if s != 0]
        self.prev_state = worker.state
        print(worker.state, inv_acts, worker.get_invalid_actions())
        assert inv_acts == worker.get_invalid_actions()
        return self.sample_action(), {}

    def on_step(self, worker: WorkerRun) -> InfoType:
        if not worker.done:
            inv_acts = [i for i, s in enumerate(worker.state) if s != 0]
            print(worker.state, inv_acts, worker.get_invalid_actions())
            assert inv_acts == worker.get_invalid_actions()
        print(self.prev_state, worker.state)
        assert self.prev_state != worker.state
        return {}


def test_invalid_actions():
    env_config = srl.EnvConfig("OX")
    rl_config = DummyRLConfig()
    context = RunContext(env_config, rl_config)

    env = srl.make_env("OX")
    worker = WorkerRun(StubWorker(rl_config), env)

    context.max_episodes = 10
    context.training = False
    context.disable_trainer = True
    play(
        context,
        env,
        DummyRLParameter(),
        DummyRLMemory(),
        workers=[worker, srl.make_worker_rulebase("random", env)],
    )

import srl
from srl.base.context import RunContext
from srl.base.define import RLActionType
from srl.base.rl.config import DummyRLConfig
from srl.base.rl.worker import RLWorker
from srl.base.rl.worker_run import WorkerRun
from srl.base.run.core_play import play


class StubWorker(RLWorker):
    def on_reset(self, worker):
        inv_acts = [i for i, s in enumerate(worker.state) if s != 0]
        assert inv_acts == worker.invalid_actions

    def policy(self, worker) -> RLActionType:
        inv_acts = [i for i, s in enumerate(worker.state) if s != 0]
        print(worker.state, inv_acts, worker.invalid_actions)
        assert inv_acts == worker.invalid_actions
        return self.sample_action()

    def on_step(self, worker):
        if not worker.done:
            inv_acts = [i for i, s in enumerate(worker.state) if s != 0]
            print(worker.state, inv_acts, worker.invalid_actions)
            assert inv_acts == worker.invalid_actions
        print(worker.state, worker.next_state)
        assert worker.state != worker.next_state


def test_invalid_actions():
    env_config = srl.EnvConfig("OX")
    rl_config = DummyRLConfig()
    context = RunContext(env_config, rl_config)

    env = env_config.make()
    worker = WorkerRun(StubWorker(rl_config), env)

    context.max_episodes = 10
    context.training = False
    context.disable_trainer = True
    play(
        context,
        env,
        workers=[worker, srl.make_worker("random", env)],
        main_worker_idx=0,
        trainer=None,
    )

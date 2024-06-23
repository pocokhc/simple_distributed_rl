import srl
from srl.base.context import RunContext


def test_raw():
    env = srl.make_env("Grid")
    worker = srl.make_worker("random", env)

    context = RunContext()
    env.setup(context)
    worker.on_start(context)

    env.reset()
    worker.on_reset(0)
    while not env.done:
        env.step(worker.policy())
        worker.on_step()

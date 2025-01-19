import srl
from srl.base.context import RunContext


def test_raw():
    env = srl.make_env("Grid")
    worker = srl.make_worker("random", env)

    context = RunContext()
    env.setup(context)
    worker.setup(context)

    env.reset()
    worker.reset(0)
    while not env.done:
        env.step(worker.policy())
        worker.on_step()

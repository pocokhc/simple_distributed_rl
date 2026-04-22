import io

import srl
from srl.base.context import RunContext


def test_raw(monkeypatch):
    # 標準入力をモック
    monkeypatch.setattr("sys.stdin", io.StringIO("3\n3\n2\n2\n2\n"))

    env = srl.make_env("EasyGrid")
    worker = srl.make_worker("human", env)

    context = RunContext(rl_cached_render_modes={"terminal"})
    env.setup(context, render_mode="terminal")
    worker.setup(context)

    env.reset()
    worker.reset(0)
    env.render()
    while not env.done:
        env.step(worker.policy())
        env.render()
        worker.on_step()

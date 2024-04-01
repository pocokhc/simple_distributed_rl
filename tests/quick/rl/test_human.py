import io

import srl
from srl.base.context import RunContext


def test_raw(monkeypatch):
    # 標準入力をモック
    monkeypatch.setattr("sys.stdin", io.StringIO("3\n3\n2\n2\n2\n"))

    env = srl.make_env("EasyGrid")
    worker = srl.make_worker_rulebase("human", env)

    context = RunContext(render_mode="terminal")
    env.setup(context)
    worker.on_start(context)

    env.reset()
    worker.on_reset(0)
    env.render()
    while not env.done:
        env.step(worker.policy())
        env.render()
        worker.on_step()

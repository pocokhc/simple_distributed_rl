import io

import srl


def test_raw(monkeypatch):
    # 標準入力をモック
    monkeypatch.setattr("sys.stdin", io.StringIO("3\n3\n2\n2\n2\n"))

    env = srl.make_env("EasyGrid")
    worker = srl.make_worker_rulebase("human", env)

    env.reset(render_mode="terminal")
    worker.on_reset(0, training=False)
    env.render()
    while not env.done:
        env.step(worker.policy())
        env.render()
        worker.on_step()

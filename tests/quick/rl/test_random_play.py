import srl


def test_raw():
    env = srl.make_env("Grid")
    worker = srl.make_worker_rulebase("random", env)

    env.reset()
    worker.on_reset(0, training=False)

    while not env.done:
        env.step(worker.policy())
        worker.on_step()

from srl import runner
from srl.envs import grid  # noqa F401

runner.run_actor("127.0.0.1", port=50000)

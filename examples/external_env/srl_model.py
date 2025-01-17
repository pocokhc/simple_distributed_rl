import env_srl  # noqa: F401 , load env

import srl
from srl.algorithms import ql


def create_runner():
    env_config = srl.EnvConfig("ExternalEnv")
    rl_config = ql.Config()
    return srl.Runner(env_config, rl_config)

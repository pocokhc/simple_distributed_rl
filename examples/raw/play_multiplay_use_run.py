import numpy as np

import srl
import srl.rl.random_play
from srl.base.run.core import play
from srl.base.run.data import RunContext
from srl.utils import common

# --- env & algorithm load
from srl.envs import grid  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip


def main():
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()

    # make instance
    env = srl.make_env(env_config)
    parameter = srl.make_parameter(rl_config, env)
    remote_memory = srl.make_memory(rl_config)
    trainer = srl.make_trainer(rl_config, parameter, remote_memory)
    worker = srl.make_worker(rl_config, env, parameter, remote_memory)

    # --- train
    context = RunContext()
    context.setup(env_config, rl_config)
    context.max_episodes = 1000
    context.training = True
    play(context, env, [worker], trainer)

    # --- evaluate
    context.max_episodes = 100
    context.training = False
    state = play(context, env, [worker])
    print(f"Average reward for 100 episodes: {np.mean(state.episode_rewards_list, axis=0)}")

    context.max_episodes = 1
    context.training = False
    context.render_mode = "terminal"
    play(context, env, [worker])


if __name__ == "__main__":
    common.logger_print()

    main()

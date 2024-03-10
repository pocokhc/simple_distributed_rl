import numpy as np

import srl
from srl.base.run.context import RunContext
from srl.base.run.core_play import play
from srl.utils import common

# --- env & algorithm load
from srl.envs import grid  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip


def main():
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()
    context = RunContext(env_config, rl_config)

    # --- setup
    env = srl.make_env(env_config)
    parameter = srl.make_parameter(rl_config, env)
    memory = srl.make_memory(rl_config, env)
    trainer = srl.make_trainer(rl_config, parameter, memory)
    workers = [srl.make_worker(rl_config, env, parameter, memory)]

    # --- train
    context.max_episodes = 10000
    context.training = True
    play(context, env, workers, 0, trainer)

    # --- evaluate
    context.max_episodes = 100
    context.training = False
    state = play(context, env, workers, 0)
    print(f"Average reward for 100 episodes: {np.mean(state.episode_rewards_list, axis=0)}")

    # --- render
    context.max_episodes = 1
    context.training = False
    context.render_mode = "terminal"
    state = play(context, env, workers, 0)


if __name__ == "__main__":
    common.logger_print()

    main()

import numpy as np

import srl
from srl.base.run.play import play
from srl.utils import common

# --- env & algorithm load
from srl.envs import grid  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip


def main():
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()

    # --- setup
    env = env_config.make()
    parameter = rl_config.make_parameter(env)
    memory = rl_config.make_memory(env)
    trainer = rl_config.make_trainer(parameter, memory, env)
    workers = [rl_config.make_worker(env, parameter, memory)]

    # --- train
    context = srl.RunContext(env_config, rl_config)
    context.max_episodes = 10000
    context.training = True
    play(context, env, workers, 0, trainer)

    # --- evaluate
    context = srl.RunContext(env_config, rl_config)
    context.max_episodes = 100
    state = play(context, env, workers, 0)
    print(f"Average reward for 100 episodes: {np.mean(state.episode_rewards_list, axis=0)}")

    # --- render
    context = srl.RunContext(env_config, rl_config)
    context.max_episodes = 1
    context.render_mode = "terminal"
    state = play(context, env, workers, 0)


if __name__ == "__main__":
    common.logger_print()

    main()

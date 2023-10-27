import numpy as np

import srl
from srl.base.run.context import RunContext
from srl.base.run.core import play
from srl.utils import common

# --- env & algorithm load
from srl.envs import grid  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip


def main():
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()
    context = RunContext(env_config, rl_config)

    # --- setup
    controller = context.create_controller()
    env = controller.make_env()
    parameter = srl.make_parameter(rl_config)
    memory = srl.make_memory(rl_config)
    controller.setup()

    # --- train
    context.max_train_count = 1000
    context.training = True
    play(context, env, parameter, memory)

    # --- evaluate
    context.max_episodes = 100
    context.training = False
    state = play(context, env, parameter, memory)
    print(f"Average reward for 100 episodes: {np.mean(state.episode_rewards_list, axis=0)}")

    # --- render
    context.max_episodes = 1
    context.training = False
    context.render_mode = "terminal"
    play(context, env, parameter, memory)


if __name__ == "__main__":
    common.logger_print()

    main()

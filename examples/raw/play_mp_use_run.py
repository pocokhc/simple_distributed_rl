import numpy as np

import srl
from srl.base.context import RunContext
from srl.base.run.play import play
from srl.base.run.play_mp import MpConfig, train
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

    # --- train
    mp_data = MpConfig(RunContext(env_config, rl_config, max_train_count=10_000))
    train(mp_data, parameter, memory)

    # --- evaluate
    workers = [srl.make_worker(rl_config, env, parameter, memory)]
    context = RunContext()
    context.max_episodes = 100
    state = play(context, env, workers, 0)
    print(f"Average reward for 100 episodes: {np.mean(state.episode_rewards_list, axis=0)}")

    # --- render
    context = RunContext(render_mode="terminal")
    context.max_episodes = 1
    state = play(context, env, workers, 0)


if __name__ == "__main__":
    common.logger_print()

    main()

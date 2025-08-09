import numpy as np

import srl
from srl.base.run.callback import RunCallback
from srl.base.run.core_play import play
from srl.utils import common

# --- env & algorithm load
from srl.envs import grid  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip


common.logger_print()


class RenderCallbacks(RunCallback):
    def on_start(self, context: srl.RunContext, **kwargs) -> None:
        context.env_render_mode = "terminal"
        context.rl_render_mode = "terminal"

    def on_step_action_after(self, context: srl.RunContext, state, **kwargs) -> None:
        state.env.render()
        state.worker.render()

    def on_episode_end(self, context: srl.RunContext, state, **kwargs) -> None:
        state.env.render()


def main():
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()

    # instance
    env = env_config.make()
    rl_config.setup(env)
    parameter = rl_config.make_parameter()
    memory = rl_config.make_memory()
    trainer = rl_config.make_trainer(parameter, memory)
    worker = rl_config.make_worker(env, parameter, memory)

    # --- train
    context = srl.RunContext(env_config, rl_config)
    context.max_episodes = 10000
    context.training = True
    state = play(context, env, worker, trainer)

    # --- evaluate
    context = srl.RunContext(env_config, rl_config)
    context.max_episodes = 100
    state = play(context, env, worker)
    print(f"Average reward for 100 episodes: {np.mean(state.episode_rewards_list, axis=0)}")

    # --- render
    context = srl.RunContext(env_config, rl_config)
    context.max_episodes = 1
    context.callbacks = [RenderCallbacks()]
    play(context, env, worker)


if __name__ == "__main__":
    main()

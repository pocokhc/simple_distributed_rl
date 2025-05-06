import numpy as np

import srl
from srl.base.run.callback import RunCallback
from srl.base.run.core_play import play
from srl.utils import common

# --- env & algorithm load
from srl.envs import grid  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip


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

    # --- train
    context = srl.RunContext(env_config, rl_config)
    context.max_episodes = 10000
    context.training = True
    state = play(context)

    # --- evaluate
    context = srl.RunContext(env_config, rl_config)
    context.max_episodes = 100
    state = play(context, state)
    print(f"Average reward for 100 episodes: {np.mean(state.episode_rewards_list, axis=0)}")

    # --- render
    context = srl.RunContext(env_config, rl_config)
    context.max_episodes = 1
    play(context, state, callbacks=[RenderCallbacks()])


if __name__ == "__main__":
    common.logger_print()

    main()

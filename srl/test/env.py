import numpy as np
import srl
from srl.base.define import EnvActionType, RenderType
from srl.base.env.base import EnvBase
from srl.runner import sequence
from srl.runner.callbacks import PrintProgress


class TestEnv:
    def __init__(self):
        pass

    def play_test(self, env_name: str):
        env = srl.envs.make_env(env_name)
        assert issubclass(env.__class__, EnvBase)

        player_num = env.player_num
        assert player_num > 0

        # --- reset
        state, next_player_indices = env.reset()
        assert isinstance(state, np.ndarray)
        for i in next_player_indices:
            assert 0 <= i < player_num

        # --- episode
        done = False
        step = 0

        # render
        for mode in RenderType:
            try:
                env.render(mode)
            except NotImplementedError:
                pass

        while not done:

            # --- sample
            actions = env.sample(next_player_indices)
            assert len(actions) == len(next_player_indices)
            if env.action_type == EnvActionType.DISCRETE:
                for i, idx in enumerate(next_player_indices):
                    action = actions[i]

                    # fetch_invalid_actions
                    invalid_actions = env.fetch_invalid_actions(idx)
                    assert isinstance(invalid_actions, list)
                    for a in invalid_actions:
                        assert isinstance(a, int)
                    assert isinstance(action, int)
                    assert 0 <= action < env.action_space.n
                    assert action not in invalid_actions

            # --- step
            state, rewards, done, next_player_indices, info = env.step(actions)
            assert len(rewards) == player_num
            assert isinstance(state, np.ndarray)
            assert isinstance(done, bool)
            assert isinstance(info, dict)
            for i in next_player_indices:
                assert 0 <= i < player_num
            # uniq check
            assert len(next_player_indices) == len(list(set(next_player_indices)))
            for reward in rewards:
                assert type(reward) in [int, float]

            step += 1
            print(f"step {step}, actions {actions}, rewards {rewards}")

            # render
            for mode in RenderType:
                try:
                    env.render(mode)
                except NotImplementedError:
                    pass

    def player_test(self, env_name: str, player: str):
        env_config = srl.envs.Config(env_name)
        rl_config = srl.rl.random_play.Config()

        config = sequence.Config(env_config, rl_config)
        env = config.make_env()
        config.players = [player] * env.player_num

        config.set_play_config(max_episodes=10, callbacks=[PrintProgress()])
        sequence.play(config)

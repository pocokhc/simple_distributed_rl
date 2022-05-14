import numpy as np
import srl
from srl.base.define import RenderType
from srl.base.env.base import EnvBase
from srl.runner import sequence
from srl.runner.callbacks import PrintProgress


class TestEnv:
    def play_test(
        self,
        env_name: str,
        check_render: bool = True,
        check_restore: bool = True,
        max_step: int = 0,
        print_disable: bool = False,
    ) -> EnvBase:

        # renderとrestoreの同時は想定しないとする
        env = self._play_test(env_name, False, check_restore, max_step, print_disable)
        if check_render:
            env = self._play_test(env_name, check_render, False, max_step, print_disable)

        return env

    def _play_test(
        self,
        env_name,
        check_render,
        check_restore,
        max_step,
        print_disable,
    ):
        env = srl.envs.make(env_name)
        assert issubclass(env.__class__, EnvBase)

        player_num = env.player_num
        assert player_num > 0

        # --- reset
        state, next_player_indices = env.reset()
        assert isinstance(state, np.ndarray)
        for i in next_player_indices:
            assert 0 <= i < player_num

        # --- restore/backup
        if check_restore:
            dat = env.backup()
            env.restore(dat)

        # --- episode
        done = False
        step = 0

        # render
        if check_render:
            for mode in RenderType:
                try:
                    env.render(mode)
                except NotImplementedError:
                    pass

        while not done:

            # --- sample
            actions = env.sample(next_player_indices)
            assert len(actions) == len(next_player_indices)

            # get_invalid_actions
            for idx in range(env.player_num):
                invalid_actions = env.get_invalid_actions(idx)
                assert isinstance(invalid_actions, list)
                for a in invalid_actions:
                    assert isinstance(a, int)

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

            if step > env.max_episode_steps:
                done = True

            if not print_disable:
                print(f"step {step}, actions {actions}, rewards {rewards}")

            # --- restore/backup
            if check_restore:
                dat = env.backup()
                env.restore(dat)

            # render
            if check_render:
                for mode in RenderType:
                    try:
                        env.render(mode)
                    except NotImplementedError:
                        pass

            if max_step > 0 and step > max_step:
                break

        env.close()
        return env

    def player_test(self, env_name: str, player: str) -> EnvBase:
        env_config = srl.envs.Config(env_name)
        rl_config = srl.rl.random_play.Config()

        config = sequence.Config(env_config, rl_config)
        env = config.make_env()
        config.players = [player] * env.player_num

        config.set_play_config(max_episodes=10, callbacks=[PrintProgress()])
        sequence.play(config)
        return env

import numpy as np
import srl
from srl import runner
from srl.base.env.base import EnvRun


class TestEnv:
    def play_test(
        self,
        env_name: str,
        check_render: bool = True,
        check_restore: bool = True,
        max_step: int = 0,
        print_enable: bool = False,
    ) -> EnvRun:

        # renderとrestoreの同時は想定しないとする
        env = self._play_test(env_name, False, check_restore, max_step, print_enable)
        if check_render:
            env = self._play_test(env_name, check_render, False, max_step, print_enable)

        return env

    def _is_space_base_instance(self, val):
        if type(val) in [int, float, list, np.ndarray]:
            return True
        return False

    def _play_test(
        self,
        env_name,
        check_render,
        check_restore,
        max_step,
        print_enable,
    ):
        env = srl.make_env(env_name)
        assert issubclass(env.__class__, EnvRun)

        player_num = env.player_num
        assert player_num > 0

        # --- reset
        env.reset()
        assert self._is_space_base_instance(env.state)
        assert 0 <= env.next_player_index < player_num

        # --- restore/backup
        if check_restore:
            dat = env.backup()
            env.restore(dat)

        assert not env.done
        assert env.step_num == 0

        # render
        if check_render:
            env.render()
            try:
                env.render_rgb_array()
            except NotImplementedError:
                pass
            try:
                env.render_terminal()
            except NotImplementedError:
                pass
            try:
                env.render_window()
            except NotImplementedError:
                pass

        while not env.done:

            # --- sample
            action = env.sample()

            # get_invalid_actions
            for idx in range(env.player_num):
                invalid_actions = env.get_invalid_actions(idx)
                assert isinstance(invalid_actions, list)
                for a in invalid_actions:
                    assert isinstance(a, int)

            # actionが選べるか
            invalid_actions = env.get_invalid_actions(env.next_player_index)
            if len(invalid_actions) > 0:
                assert len(invalid_actions) < env.action_space.get_action_discrete_info()

            # --- step
            env.step(action)
            assert self._is_space_base_instance(env.state)
            assert isinstance(env.done, bool)
            assert isinstance(env.info, dict)
            assert 0 <= env.next_player_index < player_num
            assert len(env.step_rewards) == player_num
            assert env.step_num > 0

            if print_enable:
                print(f"step {env.step_num}, actions {action}, rewards {env.step_rewards}")

            # --- restore/backup
            if check_restore:
                dat = env.backup()
                env.restore(dat)

            # render
            if check_render:
                env.render()
                try:
                    env.render_rgb_array()
                except NotImplementedError:
                    pass
                try:
                    env.render_terminal()
                except NotImplementedError:
                    pass
                try:
                    env.render_window()
                except NotImplementedError:
                    pass

            if max_step > 0 and env.step_num > max_step:
                break

        env.close()
        return env

    def player_test(self, env_name: str, player: str) -> EnvRun:
        env_config = srl.EnvConfig(env_name)
        config = runner.Config(env_config, None)

        env = config.make_env()
        config.players = [player] * env.player_num

        runner.evaluate(config, None, max_episodes=10)
        return env

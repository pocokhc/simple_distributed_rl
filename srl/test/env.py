import numpy as np
import srl
from srl.base.define import RenderType
from srl.base.env.base import EnvRun
from srl.runner import sequence
from srl.runner.callbacks import PrintProgress


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
        env = srl.envs.make(env_name)
        assert issubclass(env.__class__, EnvRun)

        player_num = env.player_num
        assert player_num > 0

        # --- reset
        env.reset()
        assert self._is_space_base_instance(env.state)
        for i in env.next_player_indices:
            assert 0 <= i < player_num

        # --- restore/backup
        if check_restore:
            dat = env.backup()
            env.restore(dat)

        assert not env.done
        assert env.step_num == 0

        # render
        if check_render:
            for mode in RenderType:
                try:
                    env.render(mode)
                except NotImplementedError:
                    pass

        while not env.done:

            # --- sample
            actions = env.samples()
            assert len(actions) == env.player_num

            # get_invalid_actions
            for idx in range(env.player_num):
                invalid_actions = env.get_invalid_actions(idx)
                assert isinstance(invalid_actions, list)
                for a in invalid_actions:
                    assert isinstance(a, int)

            # --- step
            env.step(actions)
            assert self._is_space_base_instance(env.state)
            assert isinstance(env.done, bool)
            assert isinstance(env.info, dict)
            for i in env.next_player_indices:
                assert 0 <= i < player_num
            # uniq check
            assert len(env.next_player_indices) == len(list(set(env.next_player_indices)))
            assert len(env.step_rewards) == player_num
            assert env.step_num > 0

            if print_enable:
                print(f"step {env.step_num}, actions {actions}, rewards {env.step_rewards}")

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

            if max_step > 0 and env.step_num > max_step:
                break

        env.close()
        return env

    def player_test(self, env_name: str, player: str) -> EnvRun:
        env_config = srl.envs.Config(env_name)
        config = sequence.Config(env_config, None)

        env = config.make_env()
        config.players = [player] * env.player_num

        sequence.evaluate(config, None, max_episodes=10)
        return env

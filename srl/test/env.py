import srl
from srl import runner
from srl.base.define import PlayRenderMode
from srl.base.env.base import EnvRun
from srl.utils.common import is_package_installed


class TestEnv:
    def play_test(
        self,
        env_name: str,
        check_render: bool = True,
        check_restore: bool = True,
        max_step: int = 0,
        print_enable: bool = False,
    ) -> EnvRun:
        env = srl.make_env(env_name)
        assert issubclass(env.__class__, EnvRun), "The way env is created is wrong. (Mainly due to framework side)"

        # backup/restore と render は同時に使用しない
        # render_terminal/render_window は1エピソードで変更しない
        if check_restore:
            env.set_render_mode(PlayRenderMode.none)
            self._play_test(
                env,
                check_restore=True,
                max_step=max_step,
                print_enable=print_enable,
            )
        if check_render:
            env.set_render_mode(PlayRenderMode.terminal, interval=1)
            self._play_test(
                env,
                check_restore=False,
                max_step=max_step,
                print_enable=print_enable,
            )
            if (
                is_package_installed("cv2")
                and is_package_installed("matplotlib")
                and is_package_installed("PIL")
                and is_package_installed("pygame")
            ):
                env.set_render_mode(PlayRenderMode.window, interval=1)
                self._play_test(
                    env,
                    check_restore=False,
                    max_step=max_step,
                    print_enable=print_enable,
                )

        env.close()
        return env

    def _play_test(
        self,
        env: EnvRun,
        check_restore,
        max_step,
        print_enable,
    ):

        player_num = env.player_num
        assert player_num > 0, "player_num is greater than or equal to 1."

        # --- reset
        env.reset()
        assert env.observation_space.check_val(env.state), f"Checking observation_space failed. state={env.state}"
        assert (
            0 <= env.next_player_index < player_num
        ), f"next_player_index is out of range. (0 <= {env.next_player_index} < {player_num}) is false."
        assert isinstance(env.info, dict), "The type of info is not dict."

        # --- restore/backup
        if check_restore:
            dat = env.backup()
            env.restore(dat)

        assert not env.done, "Done should be True after reset."
        assert env.step_num == 0, "step_num should be 0 after reset."

        # render
        env.render()

        while not env.done:

            # --- sample
            action = env.sample()

            # get_invalid_actions
            for idx in range(env.player_num):
                invalid_actions = env.get_invalid_actions(idx)
                assert isinstance(invalid_actions, list), "get_invalid_actions should return a list[int] type."
                for a in invalid_actions:
                    assert isinstance(a, int), "get_invalid_actions should return a list[int] type."
                    assert env.action_space.check_val(a), f"Checking action_space failed. action={a}"

            # --- step
            env.step(action)
            assert env.observation_space.check_val(env.state), f"Checking observation_space failed. state={env.state}"
            assert isinstance(env.done, bool), "The type of done is not bool."
            assert isinstance(env.info, dict), "The type of info is not dict."
            assert (
                0 <= env.next_player_index < player_num
            ), f"next_player_index is out of range. (0 <= {env.next_player_index} < {player_num}) is false."
            assert len(env.step_rewards) == player_num, "The number of rewards and players do not match."
            assert env.step_num > 0, "steps not counted.(Mainly due to framework side)"

            if print_enable:
                print(f"step {env.step_num}, actions {action}, rewards {env.step_rewards}")

            # --- restore/backup
            if check_restore:
                dat = env.backup()
                env.restore(dat)

            # render
            env.render()

            if max_step > 0 and env.step_num > max_step:
                break

        return env

    def player_test(self, env_name: str, player: str) -> EnvRun:
        env_config = srl.EnvConfig(env_name)
        config = runner.Config(env_config, None)

        env = config.make_env()
        config.players = [player] * env.player_num

        runner.evaluate(config, None, max_episodes=10)
        return env

import srl
from srl.base.context import RunContext
from srl.base.define import RenderModes
from srl.base.env.env_run import EnvRun
from srl.utils.common import is_available_pygame_video_device, is_packages_installed


class TestEnv:
    def play_test(
        self,
        env_name: str,
        env_config_kwargs={},
        check_render: bool = True,
        check_restore: bool = True,
        max_step: int = 0,
        print_enable: bool = False,
    ) -> EnvRun:
        env_config = srl.EnvConfig(
            env_name,
            render_interval=1,
            enable_assertion=True,
            **env_config_kwargs,
        )
        env = env_config.make()
        assert issubclass(env.__class__, EnvRun), "The way env is created is wrong. (Mainly due to framework side)"

        # --- make_worker test
        worker = env.make_worker("AAAAAA", enable_raise=False)
        assert worker is None

        # ---
        # backup/restore と render は同時に使用しない
        # render_terminal/render_window は1エピソードで変更しない
        if check_restore:
            self._play_test(
                env,
                RenderModes.none,
                check_restore=True,
                max_step=max_step,
                print_enable=print_enable,
            )
        if check_render:
            self._play_test(
                env,
                RenderModes.terminal,
                check_restore=False,
                max_step=max_step,
                print_enable=print_enable,
            )
            if is_packages_installed(["cv2", "pygame"]) and is_available_pygame_video_device():
                self._play_test(
                    env,
                    RenderModes.window,
                    check_restore=False,
                    max_step=max_step,
                    print_enable=print_enable,
                )

        env.close()
        return env

    def _play_test(
        self,
        env: EnvRun,
        render_mode: RenderModes,
        check_restore: bool,
        max_step: int,
        print_enable: bool,
    ):
        player_num = env.player_num
        assert player_num > 0, "player_num is greater than or equal to 1."

        context = RunContext(render_mode=render_mode)
        env.setup(context)

        # --- reset
        env.reset()
        assert env.observation_space.check_val(env.state), f"Checking observation_space failed. state={env.state}"
        assert (
            0 <= env.next_player_index < player_num
        ), f"next_player_index is out of range. (0 <= {env.next_player_index} < {player_num}) is false."

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
            action = env.sample_action()

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
            assert len(env.step_rewards) == player_num, "The number of rewards and players do not match."
            assert env.step_num > 0, "steps not counted.(Mainly due to framework side)"
            if not env.done:
                assert (
                    0 <= env.next_player_index < player_num
                ), f"next_player_index is out of range. (0 <= {env.next_player_index} < {player_num}) is false."

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

    def player_test(
        self,
        env_name: str,
        player: str,
        player_kwargs: dict = {},
        timeout: int = -1,
    ) -> EnvRun:
        env_config = srl.EnvConfig(env_name)
        env_config.enable_assertion = True
        runner = srl.Runner(env_config, None)

        env = runner.make_env()
        runner.set_players([(player, player_kwargs) for _ in range(env.player_num)])

        runner.evaluate(max_episodes=1, timeout=timeout)
        return env

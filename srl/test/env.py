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

    def _play_test(
        self,
        env_name,
        check_render,
        check_restore,
        max_step,
        print_enable,
    ):
        env = srl.make_env(env_name)
        assert issubclass(env.__class__, EnvRun), "The way env is created is wrong. (Mainly due to framework side)"

        player_num = env.player_num
        assert player_num > 0, "player_num is greater than or equal to 1."

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

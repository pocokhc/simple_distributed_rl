from typing import Union

import srl
from srl.base.define import RenderModes
from srl.base.env.config import EnvConfig
from srl.base.env.env_run import EnvRun
from srl.utils.common import is_available_pygame_video_device, is_packages_installed


def env_test(
    env_config: Union[str, EnvConfig],
    test_render_terminal: bool = True,
    test_render_window: bool = True,
    test_restore: bool = True,
    max_step: int = 0,
    enable_print: bool = False,
):
    test_env_config: srl.EnvConfig = srl.EnvConfig(env_config) if isinstance(env_config, str) else env_config.copy()
    test_env_config.render_interval = 1
    test_env_config.enable_assertion = True

    env = test_env_config.make()
    assert issubclass(env.__class__, EnvRun), "The way env is created is wrong. (Mainly due to framework side)"

    # --- make_worker test
    worker = env.make_worker("DUMMY_WORKER_NAME", enable_raise=False)
    assert worker is None

    # backup/restore と render は同時に使用しない
    # render_terminal/render_window は1エピソードで変更しない
    if test_restore:
        _env_test(
            env,
            RenderModes.none,
            test_restore=True,
            max_step=max_step,
            enable_print=enable_print,
        )
    if test_render_terminal:
        _env_test(
            env,
            RenderModes.terminal,
            test_restore=False,
            max_step=max_step,
            enable_print=enable_print,
        )
    if test_render_window:
        assert is_packages_installed(["cv2", "pygame"])
        assert is_available_pygame_video_device()
        _env_test(
            env,
            RenderModes.window,
            test_restore=False,
            max_step=max_step,
            enable_print=enable_print,
        )

    env.close()
    return env


def _env_test(
    env: EnvRun,
    render_mode: RenderModes,
    test_restore: bool,
    max_step: int,
    enable_print: bool,
):
    player_num = env.player_num
    assert player_num > 0, "player_num is greater than or equal to 1."

    env.setup(render_mode=render_mode)

    # --- reset
    env.reset()
    assert env.observation_space.check_val(env.state), f"Checking observation_space failed. state={env.state}"
    assert 0 <= env.next_player < player_num, f"next_player_index is out of range. (0 <= {env.next_player} < {player_num}) is false."

    # --- restore/backup
    if test_restore:
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
        assert len(env.rewards) == player_num, "The number of rewards and players do not match."
        assert env.step_num > 0, "steps not counted.(Mainly due to framework side)"
        if not env.done:
            assert 0 <= env.next_player < player_num, f"next_player_index is out of range. (0 <= {env.next_player} < {player_num}) is false."

        if enable_print:
            print(f"step {env.step_num}, actions {action}, rewards {env.rewards}")

        # --- restore/backup
        if test_restore:
            dat = env.backup()
            env.restore(dat)

        # render
        env.render()

        if max_step > 0 and env.step_num > max_step:
            break

    return env


def player_test(
    env_config: Union[str, EnvConfig],
    player: str,
    player_kwargs: dict = {},
    timeout: int = -1,
) -> EnvRun:
    test_env_config: srl.EnvConfig = srl.EnvConfig(env_config) if isinstance(env_config, str) else env_config.copy()
    test_env_config.enable_assertion = True
    runner = srl.Runner(env_config, None)

    env = runner.make_env()
    players = [(player, player_kwargs) for _ in range(env.player_num)]
    runner.evaluate(max_episodes=1, timeout=timeout, players=players)
    return env

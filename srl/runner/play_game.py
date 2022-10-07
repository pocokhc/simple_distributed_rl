import logging
import traceback
from typing import List, Union

from srl.base.define import KeyBindType
from srl.base.env.config import EnvConfig
from srl.base.rl.base import RLConfig
from srl.runner.callback import GameCallback
from srl.runner.callbacks.file_log_reader import FileLogReader
from srl.utils.common import is_packages_installed

logger = logging.getLogger(__name__)


def env_play(
    env_config: EnvConfig,
    players: List[Union[None, str, RLConfig]] = [None],
    key_bind: KeyBindType = None,
    action_division_num: int = 5,
    # file_log
    enable_file_logger: bool = True,
    file_logger_tmp_dir: str = "tmp",
    file_logger_enable_episode_log: bool = False,
    file_logger_episode_log_add_render: bool = True,
    # other
    callbacks: List[GameCallback] = [],
):
    callbacks = callbacks[:]

    if not is_packages_installed(
        [
            "cv2",
            "matplotlib",
            "PIL",
            "pygame",
        ]
    ):
        assert (
            False
        ), "To use animation you need to install 'cv2', 'matplotlib', 'PIL', 'pygame'. (pip install opencv-python matplotlib pillow pygame)"

    from srl.runner.game_window import PlayableGame

    # --- FileLog
    if enable_file_logger:
        from srl.runner.callbacks.file_log_writer import FileLogWriter

        file_logger = FileLogWriter(
            tmp_dir=file_logger_tmp_dir,
            enable_train_log=False,
            enable_episode_log=file_logger_enable_episode_log,
            add_render=file_logger_episode_log_add_render,
            enable_checkpoint=False,
        )
        callbacks.append(file_logger)
    else:
        file_logger = None

    game = PlayableGame(env_config, players, key_bind, action_division_num, callbacks)
    game.play()

    # --- history
    history = FileLogReader()
    try:
        if file_logger is not None:
            history.load(file_logger.base_dir)
    except Exception:
        logger.info(traceback.format_exc())

    return history

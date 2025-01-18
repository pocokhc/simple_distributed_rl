import logging
from pathlib import Path
from typing import List, Union, cast

import numpy as np

import srl
from srl.base.env.config import EnvConfig
from srl.base.rl.config import DummyRLConfig, RLConfig
from srl.utils.common import is_available_pygame_video_device, is_packages_installed

logger = logging.getLogger(__name__)


def test_rl(
    rl_config: RLConfig,
    env_list: List[Union[str, EnvConfig]] = ["Grid", "OX"],
    use_layer_processor: bool = False,
    device: str = "AUTO",
    test_train_kwargs: dict = dict(max_train_count=2),
    test_mp: bool = False,
    test_render_terminal: bool = True,
    test_render_window: bool = True,
    test_backup: bool = True,
    tmp_dir: str = "",
):
    rl_config = rl_config.copy()
    env_list = env_list.copy()
    train_kwargs_ = {}
    train_kwargs_.update(test_train_kwargs)

    tmp_dir_path = Path(tmp_dir)

    # rlの型チェック機能を有効化
    rl_config.enable_assertion = True

    for env_config in env_list:
        test_rl_config = rl_config.copy(reset_env_config=True)

        # envをlayerモードに変更する
        if use_layer_processor:
            if env_config == "Grid":
                env_config = "Grid-layer"
            elif env_config == "OX":
                env_config = "OX-layer"

        test_env_config: srl.EnvConfig = srl.EnvConfig(env_config) if isinstance(env_config, str) else env_config.copy()

        # runnerを作成
        runner = srl.Runner(test_env_config, test_rl_config)
        runner.set_device(device)

        # --- test train
        if not test_mp:
            print(f"--- {test_env_config.name} sequence test start ---")
            state = runner.train(**train_kwargs_)
            assert state.trainer is not None
            if train_kwargs_.get("max_train_count", 0) > 0:
                assert state.trainer.get_train_count() > 0
        else:
            print(f"--- {test_env_config.name} mp check start ---")
            if "max_steps" in train_kwargs_:
                train_kwargs_["max_train_count"] = train_kwargs_["max_steps"]
                del train_kwargs_["max_steps"]
            runner.train_mp(actor_num=2, **train_kwargs_)

        # --- test eval
        runner.evaluate(max_episodes=2, max_steps=5)

        # --- test render
        if test_render_terminal:
            runner.render_terminal(max_steps=5)
        if test_render_window:
            assert is_packages_installed(["cv2", "PIL", "pygame"])
            if is_available_pygame_video_device():
                runner.render_window(max_steps=2, render_interval=1)
            else:
                logger.warning("render_window skip. (pygame video disable)")
            runner.animation_save_gif(str(tmp_dir_path / "_tmp.gif"), max_steps=2, render_interval=1)

        # --- test backup
        if test_backup:
            runner.save_parameter(str(tmp_dir_path / "_tmp.dat"))
            runner.load_parameter(str(tmp_dir_path / "_tmp.dat"))
            # load後に学習できること
            runner.train(**train_kwargs_)


def test_rl_rulebase(
    name: str,
    env_list: List[Union[str, EnvConfig]] = ["Grid", "OX"],
    use_layer_processor: bool = False,
    device: str = "AUTO",
    test_train_kwargs: dict = dict(
        max_train_count=-1,
        max_steps=10,
        timeout=-1,
    ),
    test_mp: bool = False,
    test_render_terminal: bool = True,
    test_render_window: bool = True,
    test_backup: bool = True,
    tmp_dir: str = "",
):
    test_rl(
        DummyRLConfig(name=name),
        env_list,
        use_layer_processor,
        device,
        test_train_kwargs,
        test_mp,
        test_render_terminal,
        test_render_window,
        test_backup,
        tmp_dir,
    )

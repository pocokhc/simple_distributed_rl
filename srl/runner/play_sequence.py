import logging
import random
import time
import traceback
from typing import List, Optional, Tuple, Union

import numpy as np
from srl.base.define import PlayRenderMode
from srl.base.env.base import EnvRun
from srl.base.rl.base import RLParameter, RLRemoteMemory
from srl.base.rl.config import RLConfig
from srl.runner.callback import Callback
from srl.runner.callbacks.file_log_reader import FileLogReader
from srl.runner.config import Config
from srl.utils.common import is_package_imported, is_package_installed, is_packages_installed

logger = logging.getLogger(__name__)


def play_facade(
    config: Config,
    # stop config
    max_episodes: int = -1,
    timeout: int = -1,
    max_steps: int = -1,
    max_train_count: int = -1,
    # play config
    shuffle_player: bool = False,
    disable_trainer: bool = False,
    seed: Optional[int] = None,
    enable_profiling: bool = True,
    # evaluate
    enable_evaluation: bool = False,
    eval_env_sharing: bool = True,
    eval_interval: int = 0,  # episode
    eval_num_episode: int = 1,
    eval_players: List[Union[None, str, RLConfig]] = [],
    # play info
    training: bool = False,
    distributed: bool = False,
    # render mode
    render_mode: Union[str, PlayRenderMode] = PlayRenderMode.none,
    render_kwargs: dict = {},
    # PrintProgress
    print_progress: bool = True,
    progress_max_time: int = 60 * 10,  # s
    progress_start_time: int = 5,
    progress_print_env_info: bool = False,
    progress_print_worker_info: bool = True,
    progress_print_train_info: bool = True,
    progress_print_worker: int = 0,
    # file_log
    enable_file_logger: bool = True,
    file_logger_tmp_dir: str = "tmp",
    file_logger_enable_train_log: bool = True,
    file_logger_train_log_interval: int = 1,  # s
    file_logger_enable_episode_log: bool = False,
    file_logger_episode_log_add_render: bool = True,
    file_logger_enable_checkpoint: bool = True,
    file_logger_checkpoint_interval: int = 60 * 20,  # s
    # Rendering
    enable_rendering: bool = False,
    step_stop: bool = False,
    use_skip_step: bool = True,
    # other
    callbacks: List[Callback] = [],
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
):
    _render_kwargs = {}
    _render_kwargs.update(render_kwargs)
    render_kwargs = _render_kwargs

    if disable_trainer:
        enable_evaluation = False  # 学習しないので
        assert (
            max_steps != -1 or max_episodes != -1 or timeout != -1
        ), "Please specify 'max_episodes', 'timeout' or 'max_steps'."
    else:
        assert (
            max_steps != -1 or max_episodes != -1 or timeout != -1 or max_train_count != -1
        ), "Please specify 'max_episodes', 'timeout' , 'max_steps' or 'max_train_count'."

    config = config.copy(env_share=True)
    # stop config
    config.max_episodes = max_episodes
    config.timeout = timeout
    config.max_steps = max_steps
    config.max_train_count = max_train_count
    # play config
    config.shuffle_player = shuffle_player
    config.disable_trainer = disable_trainer
    if config.seed is None:
        config.seed = seed
    config.enable_profiling = enable_profiling
    # callbacks
    config.callbacks = callbacks[:]
    # play info
    config.training = training
    config.distributed = distributed

    # render mode
    render_mode = PlayRenderMode.from_str(render_mode)
    config.render_mode = render_mode
    config.render_kwargs = render_kwargs
    if render_mode in [PlayRenderMode.rgb_array, PlayRenderMode.window]:
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

    # --- Evaluate(最初に追加)
    if enable_evaluation:
        from srl.runner.callbacks.evaluate import Evaluate

        config.callbacks.insert(
            0,
            Evaluate(
                env_sharing=eval_env_sharing,
                interval=eval_interval,
                num_episode=eval_num_episode,
                eval_players=eval_players,
            ),
        )

    # --- PrintProgress
    if print_progress:
        from srl.runner.callbacks.print_progress import PrintProgress

        config.callbacks.append(
            PrintProgress(
                max_time=progress_max_time,
                start_time=progress_start_time,
                print_env_info=progress_print_env_info,
                print_worker_info=progress_print_worker_info,
                print_train_info=progress_print_train_info,
                print_worker=progress_print_worker,
            )
        )

    # --- FileLog
    if enable_file_logger:
        from srl.runner.callbacks.file_log_writer import FileLogWriter

        file_logger = FileLogWriter(
            tmp_dir=file_logger_tmp_dir,
            enable_train_log=file_logger_enable_train_log,
            train_log_interval=file_logger_train_log_interval,
            enable_episode_log=file_logger_enable_episode_log,
            add_render=file_logger_episode_log_add_render,
            enable_checkpoint=file_logger_enable_checkpoint,
            checkpoint_interval=file_logger_checkpoint_interval,
        )
        config.callbacks.append(file_logger)
    else:
        file_logger = None

    # --- Rendering
    if enable_rendering:
        from srl.runner.callbacks.rendering import Rendering

        render = Rendering(
            step_stop=step_stop,
            use_skip_step=use_skip_step,
        )
        config.callbacks.append(render)
    else:
        render = None

    # --- play
    episode_rewards, parameter, memory, env = play(config, parameter, remote_memory)

    # --- history

    history = FileLogReader()
    try:
        if file_logger is not None:
            history.load(file_logger.base_dir)
    except Exception:
        logger.info(traceback.format_exc())

    return episode_rewards, parameter, memory, history, render


# pynvmlはプロセス毎に管理したい
__enabled_nvidia = False


def play(
    config: Config,
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
    actor_id: int = 0,
) -> Tuple[List[List[float]], RLParameter, RLRemoteMemory, EnvRun]:
    global __enabled_nvidia

    # --- init profile
    initialized_nvidia = False
    if actor_id == 0 and config.enable_profiling:
        config.enable_ps = is_package_installed("psutil")
        if not __enabled_nvidia:
            config.enable_nvidia = False
            if is_package_installed("pynvml"):
                import pynvml

                try:
                    pynvml.nvmlInit()
                    config.enable_nvidia = True
                    __enabled_nvidia = True
                    initialized_nvidia = True
                except Exception:
                    logger.info(traceback.format_exc())

    # --- random seed
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)

        if is_package_imported("tensorflow"):
            import tensorflow as tf

            tf.random.set_seed(config.seed)

    # --- create env
    env = config.make_env()
    if config.seed is not None:
        env.set_seed(config.seed)

    # --- config
    config = config.copy(env_share=True)
    config.assert_params()

    # --- parameter/remote_memory/trainer
    if parameter is None:
        parameter = config.make_parameter()
    if remote_memory is None:
        remote_memory = config.make_remote_memory()
    if config.training and not config.disable_trainer:
        trainer = config.make_trainer(parameter, remote_memory)
    else:
        trainer = None

    # callbacks
    callbacks = [c for c in config.callbacks if issubclass(c.__class__, Callback)]

    # --- workers
    workers = [config.make_player(i, parameter, remote_memory, actor_id) for i in range(env.player_num)]

    # callbacks
    _info = {
        "config": config,
        "env": env,
        "parameter": parameter,
        "remote_memory": remote_memory,
        "trainer": trainer,
        "workers": workers,
        "actor_id": actor_id,
    }
    [c.on_episodes_begin(_info) for c in callbacks]

    # --- rewards
    episode_rewards_list = []

    logger.debug(f"timeout          : {config.timeout}s")
    logger.debug(f"max_steps        : {config.max_steps}")
    logger.debug(f"max_episodes     : {config.max_episodes}")
    logger.debug(f"max_train_count  : {config.max_train_count}")
    logger.debug(f"players          : {config.players}")

    # --- render init
    env.set_render_mode(config.render_mode)
    [w.set_render_mode(config.render_mode) for w in workers]

    # --- init
    episode_count = -1
    total_step = 0
    elapsed_t0 = time.time()
    worker_indices = [i for i in range(env.player_num)]
    episode_t0 = 0
    end_reason = ""
    worker_idx = 0

    # --- loop
    while True:
        _time = time.time()

        # --- stop check
        if config.timeout > 0 and (_time - elapsed_t0) > config.timeout:
            end_reason = "timeout."
            break

        if config.max_steps > 0 and total_step > config.max_steps:
            end_reason = "max_steps over."
            break

        if trainer is not None:
            if config.max_train_count > 0 and trainer.get_train_count() > config.max_train_count:
                end_reason = "max_train_count over."
                break

        # ------------------------
        # episode end / init
        # ------------------------
        if env.done:
            episode_count += 1

            if config.max_episodes > 0 and episode_count >= config.max_episodes:
                end_reason = "episode_count over."
                break  # end

            # env reset
            episode_t0 = _time
            env.reset()

            # shuffle
            if config.shuffle_player:
                random.shuffle(worker_indices)
            worker_idx = worker_indices[env.next_player_index]

            # worker reset
            [w.on_reset(env, worker_indices[i]) for i, w in enumerate(workers)]

            _info["episode_count"] = episode_count
            _info["worker_indices"] = worker_indices
            _info["worker_idx"] = worker_idx
            _info["player_index"] = env.next_player_index
            _info["action"] = None
            _info["step_time"] = 0
            _info["train_info"] = None
            _info["train_time"] = 0
            [c.on_episode_begin(_info) for c in callbacks]

        # ------------------------
        # step
        # ------------------------
        [c.on_step_action_before(_info) for c in callbacks]

        # action
        action = workers[worker_idx].policy(env)
        _info["action"] = action

        [c.on_step_begin(_info) for c in callbacks]

        # env step
        if config.env_config.frameskip == 0:
            env.step(action)
        else:
            env.step(action, lambda: [c.on_skip_step(_info) for c in callbacks])
        worker_idx = worker_indices[env.next_player_index]

        # rl step
        [w.on_step(env) for w in workers]

        # step update
        step_time = time.time() - _time
        total_step += 1

        # trainer
        if config.training and trainer is not None:
            _t0 = time.time()
            train_info = trainer.train()
            train_time = time.time() - _t0
        else:
            train_info = None
            train_time = 0

        _info["step_time"] = step_time
        _info["train_info"] = train_info
        _info["train_time"] = train_time
        [c.on_step_end(_info) for c in callbacks]
        _info["worker_idx"] = worker_idx
        _info["player_index"] = env.next_player_index

        if env.done:
            worker_rewards = [env.episode_rewards[worker_indices[i]] for i in range(env.player_num)]
            episode_rewards_list.append(worker_rewards)

            _info["episode_step"] = env.step_num
            _info["episode_rewards"] = env.episode_rewards
            _info["episode_time"] = time.time() - episode_t0
            _info["episode_count"] = episode_count
            [c.on_episode_end(_info) for c in callbacks]

        # callback end
        if True in [c.intermediate_stop(_info) for c in callbacks]:
            end_reason = "callback.intermediate_stop"
            break

    logger.debug(f"end_reason : {end_reason}")

    # 一度もepisodeを終了していない場合は例外で途中経過を保存
    if len(episode_rewards_list) == 0:
        worker_rewards = [env.episode_rewards[worker_indices[i]] for i in range(env.player_num)]
        episode_rewards_list.append(worker_rewards)

    _info["episode_count"] = episode_count
    _info["end_reason"] = end_reason
    [c.on_episodes_end(_info) for c in callbacks]

    # close profile
    if initialized_nvidia:
        config.enable_nvidia = False
        __enabled_nvidia = False
        try:
            import pynvml

            pynvml.nvmlShutdown()
        except Exception:
            logger.info(traceback.format_exc())

    return episode_rewards_list, parameter, remote_memory, env

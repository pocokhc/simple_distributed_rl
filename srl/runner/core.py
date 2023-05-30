import logging
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple, Union

from srl.base.define import PlayRenderModes
from srl.base.rl.base import RLParameter, RLRemoteMemory
from srl.base.rl.config import RLConfig
from srl.runner.callback import Callback
from srl.runner.callbacks.history_viewer import HistoryViewer
from srl.runner.config import Config
from srl.utils.common import is_enable_tf_device_name, is_package_installed, set_seed

logger = logging.getLogger(__name__)


@dataclass
class EvalOption:
    env_sharing: bool = True
    interval: int = 0  # episode
    # stop config
    episode: int = 1
    timeout: int = -1
    max_steps: int = -1
    # play config
    players: List[Union[None, str, Tuple[str, dict], RLConfig]] = field(default_factory=list)
    shuffle_player: bool = False
    # tensorflow options
    tf_disable: bool = True
    # random
    seed: Optional[int] = None
    # other
    callbacks: List[Callback] = field(default_factory=list)


@dataclass
class ProgressOption:
    interval_max_time: int = 60 * 10  # s
    print_start_time: int = 5
    print_env_info: bool = False
    print_worker_info: bool = True
    print_train_info: bool = True
    print_worker: int = 0


@dataclass
class HistoryOption:
    write_memory: bool = True
    write_file: bool = False
    log_interval: int = 1  # s


@dataclass
class CheckpointOption:
    checkpoint_interval: int = 60 * 20  # s
    eval: Optional[EvalOption] = None


class Options:
    def __init__(self) -> None:
        self.eval: Optional[EvalOption] = None
        self.progress: Optional[ProgressOption] = None
        self.history: Optional[HistoryOption] = None
        self.checkpoint: Optional[CheckpointOption] = None

    def copy(self):
        o = Options()
        o.eval = self.eval
        o.progress = self.progress
        o.history = self.history
        o.checkpoint = self.checkpoint
        return o


def play(
    config: Config,
    # stop config
    max_episodes: int = -1,
    timeout: int = -1,
    max_steps: int = -1,
    max_train_count: int = -1,
    # play config
    train_only: bool = False,
    shuffle_player: bool = False,
    disable_trainer: bool = False,
    enable_profiling: bool = True,
    # play info
    training: bool = False,
    distributed: bool = False,
    render_mode: Union[str, PlayRenderModes] = PlayRenderModes.none,
    # options
    eval: Optional[EvalOption] = None,
    progress: Optional[ProgressOption] = ProgressOption(),
    history: Optional[HistoryOption] = None,
    checkpoint: Optional[CheckpointOption] = None,
    # other
    callbacks: List[Callback] = [],
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
):  # 型アノテーションはimportしたくないので省略
    if not distributed:
        if train_only:
            disable_trainer = False
            training = True
            assert max_train_count > 0 or timeout > 0, "Please specify 'max_train_count' or 'timeout'."
        elif disable_trainer:
            assert (
                max_steps > 0 or max_episodes > 0 or timeout > 0
            ), "Please specify 'max_episodes', 'timeout' or 'max_steps'."
        else:
            assert (
                max_steps > 0 or max_episodes > 0 or timeout > 0 or max_train_count > 0
            ), "Please specify 'max_episodes', 'timeout' , 'max_steps' or 'max_train_count'."

    config.init_play()
    config = config.copy(env_share=True)

    # stop config
    config._max_episodes = max_episodes
    config._timeout = timeout
    config._max_steps = max_steps
    config._max_train_count = max_train_count
    # play config
    config._shuffle_player = shuffle_player
    config._disable_trainer = disable_trainer
    config._enable_profiling = enable_profiling
    # callbacks
    config._callbacks = callbacks[:]
    # play info
    config._training = training
    config._distributed = distributed

    # --- Evaluate(最初に追加)
    if eval is not None:
        from srl.runner.callbacks.evaluate import Evaluate

        config.callbacks.insert(0, Evaluate(**asdict(eval)))

    # --- PrintProgress
    if progress is not None:
        from srl.runner.callbacks.print_progress import PrintProgress

        config.callbacks.append(PrintProgress(**asdict(progress)))

    # --- history
    history_memory = None
    if history is not None:
        if history.write_memory:
            from srl.runner.callbacks.history_on_memory import HistoryOnMemory

            history_memory = HistoryOnMemory()
            config.callbacks.append(history_memory)

        if history.write_file:
            from srl.runner.callbacks.history_on_file import HistoryOnFile

            config.callbacks.append(
                HistoryOnFile(
                    save_dir=config.save_dir,
                    log_interval=history.log_interval,
                )
            )

    # --- checkpoint
    if checkpoint is not None:
        from srl.runner.callbacks.checkpoint import Checkpoint

        config.callbacks.append(
            Checkpoint(
                save_dir=os.path.join(config.save_dir, "params"),
                **asdict(checkpoint),
            )
        )

    # --- play
    episode_rewards, parameter, memory = _play_main_tf(
        config,
        parameter,
        remote_memory,
        render_mode=render_mode,
        train_only=train_only,
    )

    # --- history
    return_history = HistoryViewer()
    try:
        if history is not None:
            if history.write_memory:
                return_history.set_memory(config, history_memory)
            elif history.write_file:
                from srl.runner.callbacks.history_on_file import HistoryOnFile

                return_history.set_dir(config.save_dir)
    except Exception:
        import traceback

        logger.info(traceback.format_exc())

    return episode_rewards, parameter, memory, return_history


def _play_main_tf(
    config: Config,
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
    render_mode: Union[str, PlayRenderModes] = PlayRenderModes.none,
    train_only: bool = False,
) -> Tuple[List[List[float]], RLParameter, RLRemoteMemory]:
    allocate = config.used_device_tf
    if (not config.tf_disable) and config.use_tf and is_enable_tf_device_name(allocate):
        import tensorflow as tf

        if (not config.distributed) and (config.run_name == "main"):
            logger.info(f"tf.device({allocate})")

        with tf.device(allocate):  # type: ignore
            return _play_main(config, parameter, remote_memory, render_mode, train_only)

    else:
        return _play_main(config, parameter, remote_memory, render_mode, train_only)


# --- (multiprocessing)
# multiprocessing("spawn")ではプロセス毎に初期化される想定
# pynvmlはプロセス毎に管理
__enabled_nvidia = False


def _play_main(
    config: Config,
    parameter: Optional[RLParameter],
    remote_memory: Optional[RLRemoteMemory],
    render_mode: Union[str, PlayRenderModes],
    train_only: bool,
) -> Tuple[List[List[float]], RLParameter, RLRemoteMemory]:
    global __enabled_nvidia

    # --- init profile
    initialized_nvidia = False
    if config.enable_profiling:
        config._enable_psutil = is_package_installed("psutil")
        if not __enabled_nvidia:
            config._enable_nvidia = False
            if is_package_installed("pynvml"):
                try:
                    import pynvml

                    pynvml.nvmlInit()
                    config._enable_nvidia = True
                    __enabled_nvidia = True
                    initialized_nvidia = True
                except Exception as e:
                    import traceback

                    logger.debug(traceback.format_exc())
                    logger.info(e)

    # --- random seed
    set_seed(config.seed, config.seed_enable_gpu)
    episode_seed = random.randint(0, 2**16)
    if config.run_name == "main":
        logger.info(f"set_seed({config.seed})")
        logger.info(f"1st episode seed: {episode_seed}")

    if config.training and not config.distributed:
        import pprint

        logger.info(f"Training Config\n{pprint.pformat(config.to_dict())}")

    # --- parameter/remote_memory
    if parameter is None:
        parameter = config.make_parameter()
    if remote_memory is None:
        remote_memory = config.make_remote_memory()

    # callbacks
    callbacks = [c for c in config.callbacks if issubclass(c.__class__, Callback)]

    # --- run loop
    if not train_only:
        episode_rewards_list = _play_run(
            config,
            parameter,
            remote_memory,
            render_mode,
            callbacks,
            episode_seed,
        )
    else:
        _play_train_only(config, parameter, remote_memory, callbacks)
        episode_rewards_list = []

    # --- close profile
    if initialized_nvidia:
        config._enable_nvidia = False
        __enabled_nvidia = False
        try:
            import pynvml

            pynvml.nvmlShutdown()
        except Exception:
            import traceback

            logger.info(traceback.format_exc())

    return episode_rewards_list, parameter, remote_memory


def _play_run(
    config: Config,
    parameter: RLParameter,
    remote_memory: RLRemoteMemory,
    render_mode: Union[str, PlayRenderModes],
    callbacks: List[Callback],
    episode_seed: int,
):
    # --- env/workers/trainer
    env = config.make_env()
    workers = config.make_players(parameter, remote_memory)
    if config.training and not config.disable_trainer:
        trainer = config.make_trainer(parameter, remote_memory)
    else:
        trainer = None

    # callbacks
    _info = {
        "config": config,
        "env": env,
        "parameter": parameter,
        "remote_memory": remote_memory,
        "trainer": trainer,
        "workers": workers,
    }
    [c.on_episodes_begin(_info) for c in callbacks]

    # --- init
    episode_rewards_list = []
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
            env.reset(render_mode=render_mode, seed=episode_seed)
            episode_seed += 1

            # shuffle
            if config.shuffle_player:
                random.shuffle(worker_indices)
            worker_idx = worker_indices[env.next_player_index]

            # worker reset
            [w.on_reset(env, worker_indices[i], render_mode=render_mode) for i, w in enumerate(workers)]

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

            def __f():
                [c.on_skip_step(_info) for c in callbacks]

            env.step(action, __f)
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
            # rewardは学習中は不要
            if not config.training:
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

    if config.training:
        logger.info(f"training end({end_reason})")

    # rewardは学習中は不要
    if not config.training:
        # 一度もepisodeを終了していない場合は例外で途中経過を保存
        if len(episode_rewards_list) == 0:
            worker_rewards = [env.episode_rewards[worker_indices[i]] for i in range(env.player_num)]
            episode_rewards_list.append(worker_rewards)

    _info["episode_count"] = episode_count
    _info["end_reason"] = end_reason
    [c.on_episodes_end(_info) for c in callbacks]

    return episode_rewards_list


def _play_train_only(
    config: Config,
    parameter: RLParameter,
    remote_memory: RLRemoteMemory,
    callbacks: List[Callback],
):
    # --- trainer
    trainer = config.make_trainer(parameter, remote_memory)

    # callbacks
    _info = {
        "config": config,
        "parameter": parameter,
        "remote_memory": remote_memory,
        "trainer": trainer,
        "train_count": 0,
    }
    [c.on_trainer_start(_info) for c in callbacks]

    # --- init
    t0 = time.time()
    end_reason = ""
    train_count = 0

    # --- loop
    while True:
        train_t0 = time.time()

        # stop check
        if config.timeout > 0 and train_t0 - t0 > config.timeout:
            end_reason = "timeout."
            break

        if config.max_train_count > 0 and train_count > config.max_train_count:
            end_reason = "max_train_count over."
            break

        # train
        train_info = trainer.train()
        train_time = time.time() - train_t0
        train_count = trainer.get_train_count()

        # callbacks
        _info["train_info"] = train_info
        _info["train_time"] = train_time
        _info["train_count"] = train_count
        [c.on_trainer_train(_info) for c in callbacks]

        # callback end
        if True in [c.intermediate_stop(_info) for c in callbacks]:
            end_reason = "callback.intermediate_stop"
            break

    # callbacks
    _info["train_count"] = train_count
    _info["end_reason"] = end_reason
    [c.on_trainer_end(_info) for c in callbacks]

    logger.info(f"training end({end_reason})")

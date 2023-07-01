import datetime
import logging
import random
import time
from typing import Optional

import numpy as np

from srl.base.rl.base import RLParameter, RLRemoteMemory, RLTrainer
from srl.runner.config import Config
from srl.utils.common import set_seed
from srl.utils.util_str import to_str_time

logger = logging.getLogger(__name__)


def train(
    config: Config,
    # stop config
    max_episodes: int = -1,
    timeout: int = -1,
    max_steps: int = -1,
    max_train_count: int = -1,
    # play config
    shuffle_player: bool = False,
    disable_trainer: bool = False,
    # progress
    print_interval_max_time: int = 60 * 10,
    print_start_time: int = 5,
    print_env_info: bool = False,
    print_worker_info: bool = True,
    print_train_info: bool = True,
    # other
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
):
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
    config._training = True
    # play config
    config._shuffle_player = shuffle_player
    config._disable_trainer = disable_trainer

    # --- random seed
    episode_seed = None
    if config.seed is not None:
        set_seed(config.seed, config.seed_enable_gpu)
        episode_seed = random.randint(0, 2**16)
        logger.info(f"set_seed({config.seed})")
        logger.info(f"1st episode seed: {episode_seed}")

    # --- parameter/remote_memory
    if parameter is None:
        parameter = config.make_parameter()
    if remote_memory is None:
        remote_memory = config.make_remote_memory()

    # --- env/workers/trainer
    env = config.make_env()
    workers = config.make_players(parameter, remote_memory)
    if config.disable_trainer:
        trainer = None
    else:
        trainer = config.make_trainer(parameter, remote_memory)

    # --- progress
    print(
        "### env: {}, rl: {}, max episodes: {}, timeout: {}, max steps: {}, max train: {}".format(
            config.env_config.name,
            config.rl_config.getName(),
            config.max_episodes,
            to_str_time(config.timeout),
            config.max_steps,
            config.max_train_count,
        )
    )
    _time = time.time()
    progress_timeout = print_start_time
    progress_t0 = _time
    episode_t0 = _time
    episode_time = 0
    step_t0 = _time
    step_time = 0
    progress_step_count = -1
    reward = 0
    episode_step = 0
    train_info: Optional[dict] = None

    # --- init
    episode_count = -1
    total_step = 0
    elapsed_t0 = _time
    end_reason = ""
    worker_indices = [i for i in range(env.player_num)]
    worker_idx = 0

    # --- loop
    while True:
        _time = time.time()
        progress_step_count += 1

        # --- progress
        if _time - progress_t0 > progress_timeout:
            progress_t0 = _time
            progress_timeout *= 2
            if progress_timeout > print_interval_max_time:
                progress_timeout = print_interval_max_time

            if progress_step_count > 0:
                step_time = (_time - step_t0) / progress_step_count
                step_t0 = _time
                progress_step_count = 0

            _train_run_print(
                config,
                trainer,
                remote_memory,
                _time,
                elapsed_t0,
                total_step,
                step_time,
                episode_count,
                episode_time,
                reward,
                episode_step,
                env.info,
                workers[worker_idx].info,
                train_info,
                print_env_info,
                print_worker_info,
                print_train_info,
            )

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

        # --- episode end / init
        if env.done:
            episode_count += 1
            episode_time = _time - episode_t0
            episode_t0 = _time
            reward = env.episode_rewards[0]
            episode_step = env.step_num

            if config.max_episodes > 0 and episode_count >= config.max_episodes:
                end_reason = "episode_count over."
                break

            # --- reset
            env.reset(seed=episode_seed)
            if episode_seed is not None:
                episode_seed += 1

            # shuffle
            if env.player_num > 1:
                if config.shuffle_player:
                    random.shuffle(worker_indices)
                worker_idx = worker_indices[env.next_player_index]

            # worker
            [w.on_reset(worker_indices[i], training=True) for i, w in enumerate(workers)]

        # --- step
        action = workers[worker_idx].policy()
        env.step(action)
        [w.on_step() for w in workers]
        total_step += 1

        # --- train
        if trainer is not None:
            train_info = trainer.train()

    _train_run_print(
        config,
        trainer,
        remote_memory,
        _time,
        elapsed_t0,
        total_step,
        step_time,
        episode_count,
        episode_time,
        reward,
        episode_step,
        env.info,
        workers[worker_idx].info,
        train_info,
        print_env_info,
        print_worker_info,
        print_train_info,
    )

    if config.training:
        logger.info(f"training end({end_reason})")

    return parameter, remote_memory


def _train_run_print(
    config: Config,
    trainer: Optional[RLTrainer],
    remote_memory: RLRemoteMemory,
    _time,
    elapsed_t0,
    total_step,
    step_time,
    episode_count,
    episode_time,
    reward,
    episode_step,
    env_info,
    worker_info,
    train_info,
    print_env_info,
    print_worker_info,
    print_train_info,
):
    # [TIME]
    s = datetime.datetime.now().strftime("%H:%M:%S")
    s += f" {to_str_time(_time - elapsed_t0)}"

    # [remain]
    if (config.max_steps > 0) and (total_step > 0):
        remain_step = (config.max_steps - total_step) * step_time
    else:
        remain_step = np.inf
    if (config.max_episodes > 0) and (episode_count > 0):
        remain_episode = (config.max_episodes - episode_count) * episode_time
    else:
        remain_episode = np.inf
    if config.timeout > 0:
        remain_time = config.timeout - (_time - elapsed_t0)
    else:
        remain_time = np.inf
    if (trainer is not None) and (config.max_train_count > 0) and (trainer.get_train_count() > 0):
        remain_train = (config.max_train_count - trainer.get_train_count()) * step_time
    else:
        remain_train = np.inf
    remain = min(min(min(remain_step, remain_episode), remain_time), remain_train)
    if remain == np.inf:
        s += "(     - left)"
    else:
        s += f"({to_str_time(remain)} left)"

    # [all step] [all episode] [train]
    s += f" {total_step:5d}st({episode_count:5d}ep)"
    if trainer is not None:
        s += " {:5d}tr".format(trainer.get_train_count())

    # [reward]
    s += f", {reward:.1f} re"

    # [episode step] [step time] [episode time]
    s += f", {episode_step:4d} step"
    s += f", {step_time:.3f}s/step"
    s += f", {episode_time:.2f}s/ep"

    # [memory]
    s += f", {remote_memory.length():5d}mem"

    # [info]
    if print_env_info and env_info is not None:
        for k, v in env_info.items():
            if v is None:
                continue
            if isinstance(v, float):
                s += f"|{k} {v:.3f}"
            else:
                s += f"|{k} {v}"
    if print_worker_info and worker_info is not None:
        for k, v in worker_info.items():
            if v is None:
                continue
            if isinstance(v, float):
                s += f"|{k} {v:.3f}"
            else:
                s += f"|{k} {v}"
    if print_train_info and train_info is not None:
        for k, v in train_info.items():
            if v is None:
                continue
            if isinstance(v, float):
                s += f"|{k} {v:.3f}"
            else:
                s += f"|{k} {v}"
    print(s)


def train_only(
    config: Config,
    remote_memory: Optional[RLRemoteMemory] = None,
    # stop config
    max_train_count: int = -1,
    timeout: int = -1,
    # progress
    print_interval_max_time: int = 60 * 10,
    print_start_time: int = 5,
    print_train_info: bool = True,
    # other
    parameter: Optional[RLParameter] = None,
):
    assert max_train_count > 0 or timeout > 0, "Please specify 'max_train_count' or 'timeout'."

    config.init_play()
    config = config.copy(env_share=True)

    # stop config
    config._timeout = timeout
    config._max_train_count = max_train_count
    config._training = True

    # --- random seed
    if config.seed is not None:
        set_seed(config.seed, config.seed_enable_gpu)
        logger.info(f"set_seed({config.seed})")

    # --- parameter/remote_memory
    if parameter is None:
        parameter = config.make_parameter()
    if remote_memory is None:
        remote_memory = config.make_remote_memory()

    # --- trainer
    trainer = config.make_trainer(parameter, remote_memory)

    # --- progress
    print(
        "### max train: {}, timeout: {}, memory len: {}".format(
            config.max_train_count,
            to_str_time(config.timeout),
            remote_memory.length(),
        )
    )
    _time = time.time()
    progress_timeout = print_start_time
    progress_t0 = _time
    train_time = 0
    train_t0 = _time
    progress_train_count = 0
    train_info: Optional[dict] = None

    # --- init
    t0 = _time
    elapsed_t0 = _time
    end_reason = ""

    # --- loop
    while True:
        _time = time.time()

        # --- progress
        if _time - progress_t0 > progress_timeout:
            progress_t0 = _time
            progress_timeout *= 2
            if progress_timeout > print_interval_max_time:
                progress_timeout = print_interval_max_time

            if progress_train_count > 0:
                train_time = (_time - train_t0) / progress_train_count
                train_t0 = _time
                progress_train_count = 0

            _train_only_print(
                config,
                trainer,
                _time,
                elapsed_t0,
                train_time,
                train_info,
                print_train_info,
            )

        # --- stop check
        if config.timeout > 0 and _time - t0 > config.timeout:
            end_reason = "timeout."
            break

        if config.max_train_count > 0 and trainer.get_train_count() > config.max_train_count:
            end_reason = "max_train_count over."
            break

        # --- train
        train_info = trainer.train()
        progress_train_count += 1

    _train_only_print(
        config,
        trainer,
        _time,
        elapsed_t0,
        train_time,
        train_info,
        print_train_info,
    )

    logger.info(f"training end({end_reason})")
    return parameter, remote_memory


def _train_only_print(
    config: Config,
    trainer: RLTrainer,
    _time,
    elapsed_t0,
    train_time,
    train_info,
    print_train_info,
):
    # [TIME]
    s = datetime.datetime.now().strftime("%H:%M:%S")
    s += f" {to_str_time(_time - elapsed_t0)}"

    # [remain]
    if config.timeout > 0:
        remain_time = config.timeout - (_time - elapsed_t0)
    else:
        remain_time = np.inf
    if (config.max_train_count > 0) and (trainer.get_train_count() > 0):
        remain_train = (config.max_train_count - trainer.get_train_count()) * train_time
    else:
        remain_train = np.inf
    remain = min(remain_time, remain_train)
    if remain == np.inf:
        s += "(     - left)"
    else:
        s += f"({to_str_time(remain)} left)"

    # [train count]
    s += " {:6d}tr".format(trainer.get_train_count())

    # [train time]
    s += f", {train_time:.1f}s/tr"

    # [info]
    if print_train_info and train_info is not None:
        for k, v in train_info.items():
            if v is None:
                continue
            if isinstance(v, float):
                s += f"|{k} {v:.3f}"
            else:
                s += f"|{k} {v}"
    print(s)

import logging
import random
import time
from typing import List, Optional, Tuple, cast

import numpy as np
from srl.base.env.base import EnvRun
from srl.base.rl.base import RLParameter, RLRemoteMemory
from srl.runner.callback import Callback
from srl.runner.config import Config
from srl.utils.common import is_package_installed

logger = logging.getLogger(__name__)


def play(
    config: Config,
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
    actor_id: int = 0,
) -> Tuple[List[List[float]], RLParameter, RLRemoteMemory, EnvRun]:

    # --- random seed
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)

        if is_package_installed("tensorflow"):
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
    callbacks = [c for c in config.callbacks if issubclass(c.__class__, Callback)]
    callbacks = cast(List[Callback], callbacks)

    # --- eval
    if config.enable_evaluation:
        eval_config = config.create_eval_config()
        eval_episode = 0

    # --- workers
    workers = [config.make_player(i, parameter, remote_memory, actor_id) for i in range(env.player_num)]

    # callback
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
    logger.debug(f"enable_evaluation: {config.enable_evaluation}")
    logger.debug(f"players          : {config.players}")

    # --- init
    episode_count = -1
    total_step = 0
    elapsed_t0 = time.time()
    worker_indices = [i for i in range(env.player_num)]
    episode_t0 = 0
    end_reason = ""

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
            [c.on_episode_begin(_info) for c in callbacks]

        # ------------------------
        # step
        # ------------------------
        [c.on_step_begin(_info) for c in callbacks]

        # action
        action = workers[worker_idx].policy(env)
        _info["action"] = action

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

            # eval
            eval_rewards = None
            if config.enable_evaluation:
                eval_episode += 1
                if eval_episode > config.eval_interval:
                    eval_rewards, _, _, _ = play(eval_config, parameter=parameter)
                    eval_rewards = np.mean(eval_rewards, axis=0)
                    eval_episode = 0

            _info["episode_step"] = env.step_num
            _info["episode_rewards"] = env.episode_rewards
            _info["episode_time"] = time.time() - episode_t0
            _info["episode_count"] = episode_count
            _info["eval_rewards"] = eval_rewards
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

    return episode_rewards_list, parameter, remote_memory, env

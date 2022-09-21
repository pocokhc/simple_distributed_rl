import logging
import pickle
import random
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import srl
import srl.rl.dummy
import srl.rl.human  # reservation
import srl.rl.random_play  # reservation
from srl.base.env.base import EnvRun
from srl.base.env.config import EnvConfig
from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory, RLTrainer
from srl.base.rl.registration import (
    make_parameter,
    make_remote_memory,
    make_trainer,
    make_worker,
    make_worker_rulebase,
)
from srl.base.rl.worker import WorkerRun
from srl.runner.callback import Callback, TrainerCallback
from srl.runner.callbacks.file_logger import FileLogger, FileLogPlot
from srl.runner.callbacks.print_progress import PrintProgress
from srl.runner.callbacks.rendering import Rendering
from srl.utils.common import is_package_installed

logger = logging.getLogger(__name__)


@dataclass
class Config:

    env_config: EnvConfig
    rl_config: RLConfig

    # multi player option
    players: List[Union[None, str, RLConfig]] = field(default_factory=list)

    def __post_init__(self):
        # stop config
        self.max_episodes: int = -1
        self.timeout: int = -1
        self.max_steps: int = -1
        self.max_train_count: int = -1
        # play config
        self.shuffle_player: bool = False
        self.disable_trainer = False
        self.seed: Optional[int] = None
        # evaluate option
        self.enable_evaluation: bool = False
        self.eval_interval: int = 0  # episode
        self.eval_num_episode: int = 1
        self.eval_players: List[Union[None, str, RLConfig]] = []
        self.eval_player: int = 0
        # callbacks
        self.callbacks: List[Union[Callback, TrainerCallback]] = []

        # play info
        self.training: bool = False
        self.distributed: bool = False

        if self.rl_config is None:
            self.rl_config = srl.rl.dummy.Config()

        self.rl_name = self.rl_config.getName()
        self.env = None

    # ------------------------------
    # user functions
    # ------------------------------
    def model_summary(self, **kwargs) -> RLParameter:
        self.make_env()
        parameter = self.make_parameter()
        parameter.summary(**kwargs)
        return parameter
        # TODO: plot model

    # ------------------------------
    # runner functions
    # ------------------------------
    def assert_params(self):
        self.make_env()
        self.rl_config.assert_params()

    def _set_env(self):
        if self.env is None:
            self.env = srl.make_env(self.env_config)
            self.rl_config.reset_config(self.env)

    def make_env(self) -> EnvRun:
        self._set_env()
        self.env.init()
        return self.env

    def make_parameter(self, is_load: bool = True) -> RLParameter:
        self._set_env()
        return make_parameter(self.rl_config, env=self.env, is_load=is_load)

    def make_remote_memory(self, is_load: bool = True) -> RLRemoteMemory:
        self._set_env()
        return make_remote_memory(self.rl_config, env=self.env, is_load=is_load)

    def make_trainer(self, parameter: RLParameter, remote_memory: RLRemoteMemory) -> RLTrainer:
        self._set_env()
        return make_trainer(self.rl_config, parameter, remote_memory, env=self.env)

    def make_worker(
        self,
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
        actor_id: int = 0,
    ) -> WorkerRun:
        self._set_env()
        worker = make_worker(
            self.rl_config,
            parameter,
            remote_memory,
            env=self.env,
            training=self.training,
            distributed=self.distributed,
            actor_id=actor_id,
        )
        return worker

    def make_player(
        self,
        player_index: int,
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
        actor_id: int = 0,
    ):
        env = self.make_env()

        # 設定されていない場合は 0 をrl、1以降をrandom
        if player_index < len(self.players):
            player_obj = self.players[player_index]
        elif player_index == 0:
            player_obj = None
        else:
            player_obj = "random"

        # none はベース
        if player_obj is None:
            return self.make_worker(parameter, remote_memory, actor_id)

        # 文字列はenv側またはルールベースのアルゴリズム
        if isinstance(player_obj, str):
            worker = env.make_worker(player_obj)
            if worker is not None:
                return worker
            worker = make_worker_rulebase(player_obj)
            if worker is not None:
                return worker
            assert True, f"not registered: {player_obj}"

        # RLConfigは専用のWorkerを作成
        if isinstance(player_obj, object) and issubclass(player_obj.__class__, RLConfig):
            parameter = make_parameter(self.rl_config)
            remote_memory = make_remote_memory(self.rl_config)
            worker = make_worker(
                player_obj,
                parameter,
                remote_memory,
                env=env,
                training=False,
                distributed=False,
                actor_id=actor_id,
            )
            return worker

        raise ValueError(f"unknown player: {player_obj}")

    # ------------------------------
    # other functions
    # ------------------------------
    def to_dict(self) -> dict:
        # listは1階層のみ
        conf = {}
        for k, v in self.__dict__.items():
            if v is None or type(v) in [int, float, bool, str]:
                conf[k] = v
            elif type(v) is list:
                conf[k] = [str(n) for n in v]

        conf["rl_config"] = {}
        for k, v in self.rl_config.__dict__.items():
            if v is None or type(v) in [int, float, bool, str]:
                conf["rl_config"][k] = v
            elif type(v) is list:
                conf["rl_config"][k] = [str(n) for n in v]

        conf["env_config"] = {}
        for k, v in self.env_config.__dict__.items():
            if v is None or type(v) in [int, float, bool, str]:
                conf["env_config"][k] = v
            elif type(v) is list:
                conf["env_config"][k] = [str(n) for n in v]

        return conf

    def copy(self, env_share: bool = False, callbacks_share: bool = True):
        self._set_env()

        env_config = self.env_config.copy()
        rl_config = self.rl_config.copy()
        config = Config(env_config, rl_config)

        # parameter
        for k, v in self.__dict__.items():
            if v is None or type(v) in [int, float, bool, str]:
                setattr(config, k, v)
            if type(v) is list:
                setattr(config, k, v)

        # list parameter
        config.players = []
        for player in self.players:
            if player is None:
                config.players.append(None)
            else:
                config.players.append(pickle.loads(pickle.dumps(player)))
        config.eval_players = []
        for player in self.eval_players:
            if player is None:
                config.eval_players.append(None)
            else:
                config.eval_players.append(pickle.loads(pickle.dumps(player)))

        # callback
        if callbacks_share:
            config.callbacks = self.callbacks
        else:
            config.callbacks = pickle.loads(pickle.dumps(self.callbacks))

        # env
        if env_share:
            config.env = self.env

        return config

    # ------------------------------
    # utility
    # ------------------------------
    def get_env_init_state(self, encode: bool = True) -> np.ndarray:
        env = self.make_env()
        env.reset()
        state = env.state
        if encode:
            worker = self.make_worker()
            state = worker.worker.state_encode(state, env)
        return state


def save(
    path: str, config: Config, parameter: Optional[RLParameter] = None, remote_memory: Optional[RLRemoteMemory] = None
) -> None:
    dat = [
        config,
        parameter.backup() if parameter is not None else None,
        remote_memory.backup(compress=True) if remote_memory is not None else None,
    ]
    with open(path, "wb") as f:
        pickle.dump(dat, f)


def load(path: str) -> Tuple[Config, RLParameter, RLRemoteMemory]:
    with open(path, "rb") as f:
        dat = pickle.load(f)
    config = dat[0]
    parameter = config.make_parameter()
    if dat[1] is not None:
        parameter.restore(dat[1])
    remote_memory = config.make_remote_memory()
    if dat[2] is not None:
        remote_memory.restore(dat[2])
    return config, parameter, remote_memory


# -------------------------------------------------


def train(
    config: Config,
    # stop config
    max_episodes: int = -1,
    timeout: int = -1,
    max_steps: int = -1,
    max_train_count: int = -1,
    # play config
    shuffle_player: bool = True,
    disable_trainer: bool = False,
    seed: Optional[int] = None,
    # evaluate
    enable_evaluation: bool = True,
    eval_interval: int = 0,  # episode
    eval_num_episode: int = 1,
    eval_players: List[Union[None, str, RLConfig]] = [],
    eval_player: int = 0,
    # PrintProgress
    print_progress: bool = True,
    progress_max_time: int = 60 * 10,  # s
    progress_start_time: int = 5,
    progress_print_env_info: bool = False,
    progress_print_worker_info: bool = True,
    progress_print_train_info: bool = True,
    progress_print_worker: int = 0,
    # history
    enable_file_logger: bool = True,
    file_logger_tmp_dir: str = "tmp",
    file_logger_interval: int = 1,  # s
    enable_checkpoint: bool = True,
    checkpoint_interval: int = 60 * 20,  # s
    # other
    callbacks: List[Callback] = [],
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
) -> Tuple[RLParameter, RLRemoteMemory, FileLogPlot]:
    eval_players = eval_players[:]
    callbacks = callbacks[:]

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
    # evaluate
    config.enable_evaluation = enable_evaluation
    config.eval_interval = eval_interval
    config.eval_num_episode = eval_num_episode
    config.eval_players = eval_players
    config.eval_player = eval_player
    # callbacks
    config.callbacks = callbacks
    # play info
    config.training = True
    config.distributed = False

    # PrintProgress
    if print_progress:
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

    # FileLogger
    if enable_file_logger:
        file_logger = FileLogger(
            tmp_dir=file_logger_tmp_dir,
            enable_log=True,
            log_interval=file_logger_interval,
            enable_checkpoint=enable_checkpoint,
            checkpoint_interval=checkpoint_interval,
        )
        config.callbacks.append(file_logger)
    else:
        file_logger = None

    # play
    _, parameter, memory, _ = play(config, parameter, remote_memory)

    # history
    history = FileLogPlot()
    try:
        if file_logger is not None:
            history.load(file_logger.base_dir)
    except Exception:
        logger.warning(traceback.format_exc())

    return parameter, memory, history


def evaluate(
    config: Config,
    parameter: Optional[RLParameter] = None,
    # stop reason
    max_episodes: int = 10,
    timeout: int = -1,
    max_steps: int = -1,
    # play config
    shuffle_player: bool = False,
    seed: Optional[int] = None,
    # PrintProgress
    print_progress: bool = False,
    progress_max_time: int = 60 * 5,  # s
    progress_start_time: int = 5,
    progress_print_env_info: bool = False,
    progress_print_worker_info: bool = True,
    progress_print_worker: int = 0,
    # other
    callbacks: List[Callback] = [],
    remote_memory: Optional[RLRemoteMemory] = None,
) -> Union[List[float], List[List[float]]]:  # single play , multi play
    callbacks = callbacks[:]

    assert (
        max_steps != -1 or max_episodes != -1 or timeout != -1
    ), "Please specify 'max_episodes', 'timeout' or 'max_steps'."

    config = config.copy(env_share=True)
    # stop config
    config.max_steps = max_steps
    config.max_episodes = max_episodes
    config.timeout = timeout
    # play config
    config.shuffle_player = shuffle_player
    config.disable_trainer = True
    if config.seed is None:
        config.seed = seed
    # evaluate
    config.enable_evaluation = False
    # callbacks
    config.callbacks = callbacks
    # play info
    config.training = False
    config.distributed = False

    # PrintProgress
    if print_progress:
        config.callbacks.append(
            PrintProgress(
                max_time=progress_max_time,
                start_time=progress_start_time,
                print_env_info=progress_print_env_info,
                print_worker_info=progress_print_worker_info,
                print_train_info=False,
                print_worker=progress_print_worker,
            )
        )

    # play
    episode_rewards, parameter, memory, env = play(config, parameter, remote_memory)

    if env.player_num == 1:
        return [r[0] for r in episode_rewards]
    else:
        return episode_rewards


def render(
    config: Config,
    parameter: Optional[RLParameter] = None,
    # Rendering
    render_terminal: bool = True,
    render_window: bool = False,
    render_kwargs: dict = {},
    step_stop: bool = False,
    enable_animation: bool = False,
    use_skip_step: bool = True,
    # stop config
    max_steps: int = -1,
    timeout: int = -1,
    seed: Optional[int] = None,
    # PrintProgress
    print_progress: bool = False,
    progress_max_time: int = 60 * 5,  # s
    progress_start_time: int = 5,
    progress_print_env_info: bool = False,
    progress_print_worker_info: bool = True,
    progress_print_worker: int = 0,
    # other
    callbacks: List[Callback] = [],
    remote_memory: Optional[RLRemoteMemory] = None,
) -> Tuple[List[float], Rendering]:
    callbacks = callbacks[:]
    _render_kwargs = {}
    _render_kwargs.update(render_kwargs)
    render_kwargs = _render_kwargs

    config = config.copy(env_share=True)
    # stop config
    config.max_episodes = 1
    config.timeout = timeout
    config.max_steps = max_steps
    config.max_train_count = -1
    # play config
    config.shuffle_player = False
    config.disable_trainer = True
    if config.seed is None:
        config.seed = seed
    # evaluate
    config.enable_evaluation = False
    # callbacks
    config.callbacks = callbacks
    # play info
    config.training = False
    config.distributed = False

    # PrintProgress
    if print_progress:
        config.callbacks.append(
            PrintProgress(
                max_time=progress_max_time,
                start_time=progress_start_time,
                print_env_info=progress_print_env_info,
                print_worker_info=progress_print_worker_info,
                print_train_info=False,
                print_worker=progress_print_worker,
            )
        )

    # Rendering
    render = Rendering(
        render_terminal=render_terminal,
        render_window=render_window,
        render_kwargs=render_kwargs,
        step_stop=step_stop,
        enable_animation=enable_animation,
        use_skip_step=use_skip_step,
    )
    config.callbacks.append(render)

    # play
    episode_rewards, parameter, memory, env = play(config, parameter, remote_memory)

    return episode_rewards[0], render


def animation(
    config: Config,
    parameter: Optional[RLParameter] = None,
    # Rendering
    render_kwargs: dict = {},
    use_skip_step: bool = True,
    # stop config
    max_steps: int = -1,
    timeout: int = -1,
    seed: Optional[int] = None,
    # PrintProgress
    print_progress: bool = False,
    progress_max_time: int = 60 * 5,  # s
    progress_start_time: int = 5,
    progress_print_env_info: bool = False,
    progress_print_worker_info: bool = True,
    progress_print_worker: int = 0,
    # other
    callbacks: List[Callback] = [],
    remote_memory: Optional[RLRemoteMemory] = None,
) -> Rendering:
    rewards, anime = render(
        config=config,
        parameter=parameter,
        render_terminal=False,
        render_window=False,
        render_kwargs=render_kwargs,
        step_stop=False,
        enable_animation=True,
        use_skip_step=use_skip_step,
        max_steps=max_steps,
        timeout=timeout,
        seed=seed,
        print_progress=print_progress,
        progress_max_time=progress_max_time,
        progress_start_time=progress_start_time,
        progress_print_env_info=progress_print_env_info,
        progress_print_worker_info=progress_print_worker_info,
        progress_print_worker=progress_print_worker,
        callbacks=callbacks,
        remote_memory=remote_memory,
    )
    return anime


# ---------------------------------
# play main
# ---------------------------------
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
        eval_config = config.copy(env_share=False)
        eval_config.enable_evaluation = False
        eval_config.players = config.eval_players
        eval_config.rl_config.remote_memory_path = ""
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

            # eval
            eval_reward = None
            if config.enable_evaluation:
                eval_episode += 1
                if eval_episode > config.eval_interval:
                    rewards = evaluate(
                        eval_config,
                        parameter=parameter,
                        max_episodes=config.eval_num_episode,
                    )
                    if env.player_num > 1:
                        rewards = [r[config.eval_player] for r in rewards]
                    eval_reward = np.mean(rewards)
                    eval_episode = 0

            _info["episode_step"] = env.step_num
            _info["episode_rewards"] = env.episode_rewards
            _info["episode_time"] = time.time() - episode_t0
            _info["episode_count"] = episode_count
            _info["eval_reward"] = eval_reward
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

import logging
import pickle
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import srl.envs
import srl.rl
from srl.base.define import RenderType
from srl.base.env.base import EnvConfig, EnvRun
from srl.base.rl.base import ExtendWorker, RLConfig, RLParameter, RLRemoteMemory, WorkerRun
from srl.base.rl.registration import (
    make_parameter,
    make_remote_memory,
    make_trainer,
    make_worker,
    make_worker_rulebase,
)
from srl.runner.callback import Callback
from srl.runner.callbacks.file_logger import FileLogger
from srl.runner.callbacks.print_progress import PrintProgress
from srl.runner.callbacks.rendering import Rendering
from srl.runner.file_log_plot import FileLogPlot

logger = logging.getLogger(__name__)


@dataclass
class Config:

    env_config: EnvConfig
    rl_config: Optional[RLConfig]
    extend_worker: Optional[Type[ExtendWorker]] = None

    # multi player option
    players: List[Union[None, str, RLConfig]] = field(default_factory=list)

    # episode option
    max_episode_steps: int = 1_000_000
    episode_timeout: int = 60 * 60  # s

    # play option
    skip_frames: int = 0

    # validation option
    validation_interval: int = 0  # episode
    validation_episode: int = 1
    validation_players: List[Union[None, str, RLConfig]] = field(default_factory=list)
    validation_player: int = 0

    # other
    is_make_env: bool = True

    def __post_init__(self):
        # play config
        self.max_steps: int = -1
        self.max_episodes: int = -1
        self.timeout: int = -1
        self.training: bool = False
        self.distributed: bool = False
        # multi player option
        self.shuffle_player: bool = False
        # validation option
        self.enable_validation: bool = False
        # callbacks
        self.callbacks: List[Callback] = []

        # none の場合はQLを代わりに入れる
        if self.rl_config is None:
            self.rl_config = srl.rl.ql.Config()

        self.rl_name = self.rl_config.getName()
        self.parameter_path = ""
        self.remote_memory_path = ""
        self.trainer_disable = False
        self.env = None

        if self.is_make_env:
            self.make_env()

    # ------------------------------
    # user functions
    # ------------------------------
    def set_parameter_path(self, parameter_path: str = "", remote_memory_path: str = ""):
        self.parameter_path = parameter_path
        self.remote_memory_path = remote_memory_path

    def model_summary(self) -> RLParameter:
        parameter = self.make_parameter()
        parameter.summary()
        return parameter

    # ------------------------------
    # runner functions
    # ------------------------------
    def assert_params(self):
        self.rl_config.assert_params()

    def make_env(self) -> EnvRun:
        if self.env is None:
            self.env = srl.envs.make(self.env_config)
            self.rl_config.set_config_by_env(self.env)
        self.env.init()
        return self.env

    def make_parameter(self) -> RLParameter:
        parameter = make_parameter(self.rl_config)
        if self.parameter_path != "":
            parameter.load(self.parameter_path)
        return parameter

    def make_remote_memory(self) -> RLRemoteMemory:
        memory = make_remote_memory(self.rl_config)
        if self.remote_memory_path != "":
            memory.load(self.remote_memory_path)
        return memory

    def make_trainer(self, parameter: RLParameter, remote_memory: RLRemoteMemory):
        return make_trainer(self.rl_config, parameter, remote_memory)

    def make_worker(
        self,
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
        actor_id: int = 0,
    ) -> WorkerRun:
        env = self.make_env()
        worker = make_worker(self.rl_config, env, parameter, remote_memory, actor_id)

        # ExtendWorker
        if self.extend_worker is not None:
            worker = self.extend_worker(worker, env)
            worker = WorkerRun(worker)

        worker.set_play_info(self.training, self.distributed)
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
        if len(self.players) == 0:
            self.players = [None]
            for _ in range(env.player_num - 1):
                self.players.append("random")

        player_obj = self.players[player_index]

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
            worker = make_worker(player_obj, env, actor_id=actor_id)
            worker.set_play_info(False, False)
            return worker

        raise ValueError(f"unknown player: {player_obj}")

    # ------------------------------
    # other functions
    # ------------------------------

    def to_dict(self) -> dict:
        conf = {}
        for k, v in self.__dict__.items():
            if type(v) in [int, float, bool, str]:
                conf[k] = v

        conf["rl_config"] = {}
        for k, v in self.rl_config.__dict__.items():
            if type(v) in [int, float, bool, str]:
                conf["rl_config"][k] = v

        return conf

    def copy(self, env_copy: bool = False):
        env_config = pickle.loads(pickle.dumps(self.env_config))
        rl_config = pickle.loads(pickle.dumps(self.rl_config))
        config = Config(env_config, rl_config, is_make_env=False)
        for k, v in self.__dict__.items():
            if type(v) in [int, float, bool, str]:
                setattr(config, k, v)
        config.players = []
        for player in self.players:
            if player is None:
                config.players.append(None)
            else:
                config.players.append(pickle.loads(pickle.dumps(player)))
        config.validation_players = []
        for player in self.validation_players:
            if player is None:
                config.validation_players.append(None)
            else:
                config.validation_players.append(pickle.loads(pickle.dumps(player)))
        config.callbacks = self.callbacks  # sync
        config.extend_worker = self.extend_worker
        if env_copy:
            config.env = self.env
        return config


def train(
    config: Config,
    # train config
    max_steps: int = -1,
    max_episodes: int = -1,
    timeout: int = -1,
    shuffle_player: bool = True,
    enable_validation: bool = True,
    # print
    print_progress: bool = True,
    max_progress_time: int = 60 * 10,  # s
    print_progress_kwargs: Optional[Dict] = None,
    # log
    enable_file_logger: bool = True,
    file_logger_kwargs: Optional[Dict] = None,
    # other
    callbacks: List[Callback] = None,
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
) -> Tuple[RLParameter, RLRemoteMemory, FileLogPlot]:
    if callbacks is None:
        callbacks = []

    config = config.copy(env_copy=True)
    config.max_steps = max_steps
    config.max_episodes = max_episodes
    config.timeout = timeout
    config.shuffle_player = shuffle_player
    config.enable_validation = enable_validation
    config.callbacks = callbacks

    config.training = True

    if print_progress:
        if print_progress_kwargs is None:
            config.callbacks.append(PrintProgress(max_progress_time=max_progress_time))
        else:
            config.callbacks.append(PrintProgress(max_progress_time=max_progress_time, **print_progress_kwargs))

    if file_logger_kwargs is None:
        logger = FileLogger()
        else:
        logger = FileLogger(**file_logger_kwargs)
    if enable_file_logger:
        config.callbacks.append(logger)

    _, parameter, memory, _ = play(config, parameter, remote_memory)

    history = FileLogPlot()
    if enable_file_logger:
    history.set_path(logger.base_dir)
    return parameter, memory, history


def evaluate(
    config: Config,
    parameter: Optional[RLParameter],
    max_episodes: int,
    max_steps: int = -1,
    timeout: int = -1,
    shuffle_player: bool = False,
    callbacks: List[Callback] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
) -> Union[List[float], List[List[float]]]:  # single play , multi play
    if callbacks is None:
        callbacks = []

    config = config.copy(env_copy=True)
    config.max_steps = max_steps
    config.max_episodes = max_episodes
    config.timeout = timeout
    config.shuffle_player = shuffle_player
    config.callbacks = callbacks

    config.enable_validation = False
    config.training = False

    episode_rewards, parameter, memory, env = play(config, parameter, remote_memory)

    if env.player_num == 1:
        return [r[0] for r in episode_rewards]
    else:
        return episode_rewards


def render(
    config: Config,
    parameter: Optional[RLParameter] = None,
    mode: Union[str, RenderType] = RenderType.Terminal,
    step_stop: bool = False,
    enable_animation: bool = False,
    max_steps: int = -1,
    timeout: int = -1,
    shuffle_player: bool = False,
    callbacks: List[Callback] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
) -> Tuple[List[float], Rendering]:
    if callbacks is None:
        callbacks = []

    config = config.copy(env_copy=True)
    config.max_steps = max_steps
    config.timeout = timeout
    config.shuffle_player = shuffle_player
    config.callbacks = callbacks

    config.enable_validation = False
    config.max_episodes = 1
    config.training = False
    config.episode_timeout = -1

    _render = Rendering(mode=mode, step_stop=step_stop, enable_animation=enable_animation)
    config.callbacks.append(_render)

    episode_rewards, parameter, memory, env = play(config, parameter, remote_memory)

    return episode_rewards[0], _render


# ---------------------------------
# play main
# ---------------------------------
def play(
    config: Config,
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
    actor_id: int = 0,
) -> Tuple[List[List[float]], RLParameter, RLRemoteMemory, EnvRun]:

    env = config.make_env()
    config = config.copy(env_copy=True)
    config.assert_params()

    if parameter is None:
        parameter = config.make_parameter()
    if remote_memory is None:
        remote_memory = config.make_remote_memory()
    if config.training and not config.trainer_disable:
        trainer = config.make_trainer(parameter, remote_memory)
    else:
        trainer = None
    callbacks = config.callbacks

    # valid
    if config.enable_validation:
        valid_config = config.copy(env_copy=False)
        valid_config.enable_validation = False
        valid_config.players = config.validation_players
        valid_episode = 0

    # workers
    workers = [config.make_player(i, parameter, remote_memory, actor_id) for i in range(env.player_num)]

    # callback
    _params = {
        "config": config,
        "env": env,
        "parameter": parameter,
        "remote_memory": remote_memory,
        "trainer": trainer,
        "workers": workers,
        "actor_id": actor_id,
    }
    [c.on_episodes_begin(**_params) for c in callbacks]

    # --- rewards
    episode_rewards_list = []

    # --- init
    episode_count = -1
    total_step = 0
    elapsed_t0 = time.time()
    worker_indices = [i for i in range(env.player_num)]
    episode_t0 = 0

    # --- loop
    while True:
        _time = time.time()

        # timeout
        elapsed_time = _time - elapsed_t0
        if config.timeout > 0 and elapsed_time > config.timeout:
            break

        # max steps
        if config.max_steps > 0 and total_step > config.max_steps:
            break

        # ------------------------
        # episode end / init
        # ------------------------
        if env.done:
            episode_count += 1

            if config.max_episodes > 0 and episode_count >= config.max_episodes:
                break  # end

            # env reset
            episode_t0 = _time
            env.reset(config.max_episode_steps, config.episode_timeout)

            # shuffle
            if config.shuffle_player:
                random.shuffle(worker_indices)

            # worker reset
            [w.on_reset(env, worker_indices[i]) for i, w in enumerate(workers)]

            # callback
            _params["episode_count"] = episode_count
            _params["worker_indices"] = worker_indices
            [c.on_episode_begin(**_params) for c in callbacks]

        # ------------------------
        # step
        # ------------------------
        # action
        actions = [w.policy(env) for w in workers]

        # callback
        _params["actions"] = actions
        [c.on_step_begin(**_params) for c in callbacks]

        # env step
        # worker -> player に並べ替え
        actions = [actions[i] for i in worker_indices]
        if config.skip_frames == 0:
            env.step(actions)
        else:
            env.step(actions, config.skip_frames, lambda: [c.on_skip_step(**_params) for c in callbacks])

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

        # callback
        _params["step_time"] = step_time
        _params["train_info"] = train_info
        _params["train_time"] = train_time
        [c.on_step_end(**_params) for c in callbacks]

        if env.done:
            episode_rewards_list.append(env.episode_rewards)

            # validation
            valid_reward = None
            if config.enable_validation:
                valid_episode += 1
                if valid_episode > config.validation_interval:
                    rewards = evaluate(valid_config, parameter=parameter, max_episodes=config.validation_episode)
                    if env.player_num > 1:
                        rewards = [r[config.validation_player] for r in rewards]
                    valid_reward = np.mean(rewards)
                    valid_episode = 0

            # callback
            _params["episode_step"] = env.step_num
            _params["episode_rewards"] = env.episode_rewards
            _params["episode_time"] = time.time() - episode_t0
            _params["episode_count"] = episode_count
            _params["valid_reward"] = valid_reward
            [c.on_episode_end(**_params) for c in callbacks]

        # callback end
        if True in [c.intermediate_stop(**_params) for c in callbacks]:
            break

    # 一度もepisodeを終了していない場合は例外で途中経過を保存
    if len(episode_rewards_list) == 0:
        episode_rewards_list.append(env.episode_rewards)

    # callback
    _params["episode_count"] = episode_count
    [c.on_episodes_end(**_params) for c in callbacks]

    return episode_rewards_list, parameter, remote_memory, env

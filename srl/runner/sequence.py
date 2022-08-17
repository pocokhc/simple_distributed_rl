import logging
import pickle
import random
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import srl.envs
import srl.rl
from srl.base.env.base import EnvConfig, EnvRun
from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory, RLTrainer, WorkerRun
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
    rl_config: RLConfig

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
        # random seed
        self.seed: Optional[int] = None

        # none の場合はQLを代わりに入れる
        if self.rl_config is None:
            self.rl_config = srl.rl.ql.Config()

        self.rl_name = self.rl_config.getName()
        self.disable_trainer = False
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
            self.env = srl.envs.make(self.env_config)
            self.rl_config.reset_config(self.env)

    def make_env(self) -> EnvRun:
        self._set_env()
        self.env.init()
        return self.env

    def make_parameter(self) -> RLParameter:
        self._set_env()
        return make_parameter(self.rl_config, env=self.env)

    def make_remote_memory(self) -> RLRemoteMemory:
        self._set_env()
        return make_remote_memory(self.rl_config, env=self.env)

    def make_trainer(self, parameter: RLParameter, remote_memory: RLRemoteMemory) -> RLTrainer:
        self._set_env()
        return make_trainer(self.rl_config, parameter, remote_memory, env=self.env)

    def make_worker(
        self,
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
        actor_id: int = 0,
    ) -> WorkerRun:
        env = self.make_env()
        worker = make_worker(self.rl_config, env, parameter, remote_memory, actor_id)
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
        if len(self.players) < env.player_num:
            self.players += ["random"] * (env.player_num - len(self.players))

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
            parameter = make_parameter(self.rl_config)
            remote_memory = make_remote_memory(self.rl_config)
            worker = make_worker(player_obj, env, parameter, remote_memory, actor_id=actor_id)
            worker.set_play_info(training=False, distributed=False)
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
        self.make_env()  # rl_config.set_config_by_env

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
        config.validation_players = []
        for player in self.validation_players:
            if player is None:
                config.validation_players.append(None)
            else:
                config.validation_players.append(pickle.loads(pickle.dumps(player)))

        # callback
        if callbacks_share:
            config.callbacks = self.callbacks
        else:
            config.callbacks = pickle.loads(pickle.dumps(self.callbacks))

        # env
        if env_share:
            config.env = self.env

        return config


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
    # train config
    max_steps: int = -1,
    max_episodes: int = -1,
    timeout: int = -1,
    shuffle_player: bool = True,
    enable_validation: bool = True,
    disable_trainer: bool = False,
    seed: Optional[int] = None,
    # print
    print_progress: bool = True,
    max_progress_time: int = 60 * 10,  # s
    print_progress_kwargs: Optional[Dict] = None,
    # log,history
    enable_file_logger: bool = True,
    file_logger_kwargs: Optional[Dict] = None,
    remove_file_logger: bool = True,
    # other
    callbacks: List[Callback] = None,
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
) -> Tuple[RLParameter, RLRemoteMemory, FileLogPlot]:
    if callbacks is None:
        callbacks = []
    if disable_trainer:
        enable_validation = False  # 学習しないので

    assert (
        max_steps != -1 or max_episodes != -1 or timeout != -1
    ), "Please specify 'max_steps', 'max_episodes' or 'timeout'."

    config = config.copy(env_share=True)
    config.max_steps = max_steps
    config.max_episodes = max_episodes
    config.timeout = timeout
    config.shuffle_player = shuffle_player
    config.enable_validation = enable_validation
    config.disable_trainer = disable_trainer
    config.callbacks = callbacks
    if config.seed is None:
        config.seed = seed

    config.training = True

    if print_progress:
        if print_progress_kwargs is None:
            print_progress_kwargs = {}
        config.callbacks.append(PrintProgress(max_progress_time=max_progress_time, **print_progress_kwargs))

    if file_logger_kwargs is None:
        file_logger = FileLogger()
    else:
        file_logger = FileLogger(**file_logger_kwargs)
    if enable_file_logger:
        config.callbacks.append(file_logger)

    _, parameter, memory, _ = play(config, parameter, remote_memory)

    try:
        history = FileLogPlot()
        if enable_file_logger:
            history.load(file_logger.base_dir, remove_file_logger)
        return parameter, memory, history
    except Exception:
        logger.warning(traceback.format_exc())

    return parameter, memory, None


def evaluate(
    config: Config,
    parameter: Optional[RLParameter] = None,
    max_episodes: int = 10,
    max_steps: int = -1,
    timeout: int = -1,
    shuffle_player: bool = False,
    seed: Optional[int] = None,
    # print
    print_progress: bool = False,
    print_progress_kwargs: Optional[Dict] = None,
    # other
    callbacks: List[Callback] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
) -> Union[List[float], List[List[float]]]:  # single play , multi play
    if callbacks is None:
        callbacks = []

    config = config.copy(env_share=True)
    config.max_steps = max_steps
    config.max_episodes = max_episodes
    config.timeout = timeout
    config.shuffle_player = shuffle_player
    config.callbacks = callbacks
    if config.seed is None:
        config.seed = seed

    config.enable_validation = False
    config.training = False

    if print_progress:
        if print_progress_kwargs is None:
            print_progress_kwargs = {}
        config.callbacks.append(PrintProgress(**print_progress_kwargs))

    episode_rewards, parameter, memory, env = play(config, parameter, remote_memory)

    if env.player_num == 1:
        return [r[0] for r in episode_rewards]
    else:
        return episode_rewards


def render(
    config: Config,
    parameter: Optional[RLParameter] = None,
    render_terminal: bool = True,
    render_window: bool = False,
    step_stop: bool = False,
    enable_animation: bool = False,
    rendering_params: dict = None,
    max_steps: int = -1,
    timeout: int = -1,
    seed: Optional[int] = None,
    # print
    print_progress: bool = False,
    print_progress_kwargs: Optional[Dict] = None,
    # other
    callbacks: List[Callback] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
) -> Tuple[List[float], Optional[Rendering]]:
    if callbacks is None:
        callbacks = []

    config = config.copy(env_share=True)
    config.max_steps = max_steps
    config.timeout = timeout
    config.shuffle_player = False
    config.callbacks = callbacks
    if config.seed is None:
        config.seed = seed

    config.enable_validation = False
    config.max_episodes = 1
    config.training = False
    config.episode_timeout = -1

    if print_progress:
        if print_progress_kwargs is None:
            print_progress_kwargs = {}
        config.callbacks.append(PrintProgress(**print_progress_kwargs))

    if rendering_params is None:
        rendering_params = {}
    _render = Rendering(
        render_terminal,
        render_window=render_window,
        enable_animation=enable_animation,
        step_stop=step_stop,
        **rendering_params,
    )
    config.callbacks.append(_render)

    episode_rewards, parameter, memory, env = play(config, parameter, remote_memory)

    return episode_rewards[0], _render


def animation(
    config: Config,
    parameter: Optional[RLParameter] = None,
    rendering_params: dict = None,
    render_terminal: bool = False,
    render_window: bool = False,
    max_steps: int = -1,
    timeout: int = -1,
    seed: Optional[int] = None,
    # print
    print_progress: bool = False,
    print_progress_kwargs: Optional[Dict] = None,
    # other
    callbacks: List[Callback] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
) -> Rendering:
    rewards, anime = render(
        config,
        parameter,
        render_terminal,
        render_window,
        step_stop=False,
        enable_animation=True,
        rendering_params=rendering_params,
        max_steps=max_steps,
        timeout=timeout,
        seed=seed,
        print_progress=print_progress,
        print_progress_kwargs=print_progress_kwargs,
        callbacks=callbacks,
        remote_memory=remote_memory,
    )
    assert anime is not None
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

    env = config.make_env()

    # --- random seed
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        env.set_seed(config.seed)

        try:
            import tensorflow as tf

            tf.random.set_seed(config.seed)
        except ImportError:
            pass

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
    callbacks = config.callbacks

    # --- valid
    if config.enable_validation:
        valid_config = config.copy(env_share=False)
        valid_config.enable_validation = False
        valid_config.players = config.validation_players
        valid_config.rl_config.remote_memory_path = ""
        valid_episode = 0

    # --- workers
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

    logger.debug(f"timeout          : {config.timeout}s")
    logger.debug(f"max_steps        : {config.max_steps}")
    logger.debug(f"max_episodes     : {config.max_episodes}")
    logger.debug(f"enable_validation: {config.enable_validation}")
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

        # timeout
        elapsed_time = _time - elapsed_t0
        if config.timeout > 0 and elapsed_time > config.timeout:
            end_reason = "timeout."
            break

        # max steps
        if config.max_steps > 0 and total_step > config.max_steps:
            end_reason = "max_steps over."
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
            env.reset(config.max_episode_steps, config.episode_timeout)

            # shuffle
            if config.shuffle_player:
                random.shuffle(worker_indices)
            worker_idx = worker_indices[env.next_player_index]

            # worker reset
            [w.on_reset(env, worker_indices[i]) for i, w in enumerate(workers)]

            _params["episode_count"] = episode_count
            _params["worker_indices"] = worker_indices
            _params["worker_idx"] = worker_idx
            _params["player_index"] = env.next_player_index
            [c.on_episode_begin(**_params) for c in callbacks]

        # ------------------------
        # step
        # ------------------------
        # action
        action = workers[worker_idx].policy(env)

        _params["action"] = action
        [c.on_step_begin(**_params) for c in callbacks]

        # env step
        if config.skip_frames == 0:
            env.step(action)
        else:
            env.step(action, config.skip_frames, lambda: [c.on_skip_step(**_params) for c in callbacks])
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

        _params["step_time"] = step_time
        _params["train_info"] = train_info
        _params["train_time"] = train_time
        [c.on_step_end(**_params) for c in callbacks]
        _params["worker_idx"] = worker_idx
        _params["player_index"] = env.next_player_index

        if env.done:
            worker_rewards = [env.episode_rewards[worker_indices[i]] for i in range(env.player_num)]
            episode_rewards_list.append(worker_rewards)

            # validation
            valid_reward = None
            if config.enable_validation:
                valid_episode += 1
                if valid_episode > config.validation_interval:
                    rewards = evaluate(
                        valid_config,
                        parameter=parameter,
                        max_episodes=config.validation_episode,
                    )
                    if env.player_num > 1:
                        rewards = [r[config.validation_player] for r in rewards]
                    valid_reward = np.mean(rewards)
                    valid_episode = 0

            _params["episode_step"] = env.step_num
            _params["episode_rewards"] = env.episode_rewards
            _params["episode_time"] = time.time() - episode_t0
            _params["episode_count"] = episode_count
            _params["valid_reward"] = valid_reward
            [c.on_episode_end(**_params) for c in callbacks]

        # callback end
        if True in [c.intermediate_stop(**_params) for c in callbacks]:
            end_reason = "callback.intermediate_stop"
            break

    logger.debug(f"end_reason : {end_reason}")

    # 一度もepisodeを終了していない場合は例外で途中経過を保存
    if len(episode_rewards_list) == 0:
        worker_rewards = [env.episode_rewards[worker_indices[i]] for i in range(env.player_num)]
        episode_rewards_list.append(worker_rewards)

    _params["episode_count"] = episode_count
    [c.on_episodes_end(**_params) for c in callbacks]

    return episode_rewards_list, parameter, remote_memory, env

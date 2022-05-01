import logging
import pickle
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import srl.envs
import srl.rl
from srl.base.env.env_for_rl import EnvConfig, EnvForRL
from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory, RLWorker
from srl.base.rl.registration import make_parameter, make_remote_memory, make_trainer, make_worker
from srl.runner.callbacks import Callback

logger = logging.getLogger(__name__)


@dataclass
class Config:

    env_config: EnvConfig
    rl_config: RLConfig

    # multi player option
    players: List[Optional[RLConfig]] = field(default_factory=list)
    shuffle_player: bool = False

    # episode option
    max_episode_steps: int = 10_000
    episode_timeout: int = 60 * 10  # s

    # play option
    max_steps: int = -1
    max_episodes: int = -1
    timeout: int = -1
    training: bool = False

    # callbacks
    callbacks: List[Callback] = field(default_factory=list)

    # other
    skip_frames: int = 0

    def __post_init__(self):
        if self.rl_config is not None:
            self.rl_name = self.rl_config.getName()
        self.parameter_path = ""
        self.remote_memory_path = ""
        self.trainer_disable = False

    # ------------------------------
    # user functions
    # ------------------------------
    def set_parameter_path(self, parameter_path: str = "", remote_memory_path: str = ""):
        self.parameter_path = parameter_path
        self.remote_memory_path = remote_memory_path

    def set_train_config(
        self,
        max_steps: int = -1,
        max_episodes: int = -1,
        timeout: int = -1,
        shuffle_player: bool = True,
        callbacks: List[Callback] = None,
    ):
        self.set_play_config(max_steps, max_episodes, timeout, True, shuffle_player, callbacks)

    def set_play_config(
        self,
        max_steps: int = -1,
        max_episodes: int = -1,
        timeout: int = -1,
        training: bool = False,
        shuffle_player: bool = False,
        callbacks: List[Callback] = None,
    ):
        if callbacks is None:
            callbacks = []
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.timeout = timeout
        self.training = training
        self.shuffle_player = shuffle_player
        self.callbacks = callbacks

    def model_summary(self) -> RLParameter:
        parameter = self.make_parameter()
        parameter.summary()
        return parameter

    # ------------------------------
    # runner functions
    # ------------------------------
    def assert_params(self):
        self.rl_config.assert_params()

    def init_rl_config(self):
        if not self.rl_config.is_set_config_by_env:
            self.make_env()

    def make_env(self) -> EnvForRL:

        env = srl.envs.make(self.env_config, self.rl_config)

        if len(self.players) == 0:
            for i in range(env.player_num):
                if i == 0:
                    self.players.append(None)
                else:
                    self.players.append(srl.rl.random_play.Config())

        return env

    def make_parameter(self) -> RLParameter:
        if not self.rl_config.is_set_config_by_env:
            self.make_env()
        parameter = make_parameter(self.rl_config, None)
        if self.parameter_path != "":
            parameter.load(self.parameter_path)
        return parameter

    def make_remote_memory(self) -> RLRemoteMemory:
        if not self.rl_config.is_set_config_by_env:
            self.make_env()
        memory = make_remote_memory(self.rl_config, None)
        if self.remote_memory_path != "":
            memory.load(self.remote_memory_path)
        return memory

    def make_trainer(self, parameter: RLParameter, remote_memory: RLRemoteMemory):
        if not self.rl_config.is_set_config_by_env:
            self.make_env()
        return make_trainer(self.rl_config, None, parameter, remote_memory)

    def make_worker(
        self,
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
        worker_id: int = 0,
    ) -> RLWorker:
        if not self.rl_config.is_set_config_by_env:
            self.make_env()
        worker = make_worker(self.rl_config, None, parameter, remote_memory, worker_id)
        worker.set_training(self.training)
        return worker

    def make_player(
        self,
        player_index: int,
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
        worker_id: int = 0,
        env: Optional[EnvForRL] = None,
    ):
        if not self.rl_config.is_set_config_by_env:
            self.make_env()
        player_obj = self.players[player_index]

        # none はベース
        if player_obj is None:
            return self.make_worker(parameter, remote_memory, worker_id)

        # 文字列はenvに登録されているアルゴリズム
        if isinstance(player_obj, str):
            if player_obj == "random":
                return make_worker(srl.rl.random_play.Config(), env)
            elif player_obj == "human":
                return make_worker(srl.rl.human.Config(), env)
            else:
                worker = env.make_worker(player_obj)
                assert worker is not None, f"not registered: {player_obj}"
                return worker

        # それ以外は専用のWorkerを作成
        if isinstance(player_obj, object) and issubclass(player_obj.__class__, RLConfig):
            worker = make_worker(player_obj, env, worker_id=worker_id)
            worker.set_training(False)
            return worker

        raise ValueError()

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

    def copy(self):
        config = Config("", None, None)
        for k, v in self.__dict__.items():
            if type(v) in [int, float, bool, str]:
                setattr(config, k, v)
        config.players = []
        for player in self.players:
            if player is None:
                config.players.append(None)
            else:
                config.players.append(pickle.loads(pickle.dumps(player)))
        config.rl_config = pickle.loads(pickle.dumps(self.rl_config))
        return config


def train(
    config: Config,
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
    env: Optional[EnvForRL] = None,
    worker_id: int = 0,
) -> Tuple[RLParameter, RLRemoteMemory]:
    episode_rewards, parameter, memory = play(config, parameter, remote_memory, env, worker_id)
    return parameter, memory


def play(
    config: Config,
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
    env: Optional[EnvForRL] = None,
    worker_id: int = 0,
) -> Union[
    Tuple[float, RLParameter, RLRemoteMemory],  # single play
    Tuple[List[float], RLParameter, RLRemoteMemory],  # multi play
]:
    # TODO config copy

    if env is None:
        env = config.make_env()
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

    # workers
    workers = [config.make_player(i, parameter, remote_memory, worker_id, env) for i in range(env.player_num)]

    # callback
    _params = {
        "config": config,
        "env": env,
        "parameter": parameter,
        "remote_memory": remote_memory,
        "trainer": trainer,
        "worker_id": worker_id,
        "workers": workers,
    }
    [c.on_episodes_begin(**_params) for c in callbacks]

    # --- rewards
    episode_rewards_list = []

    # --- init
    episode_count = -1
    total_step = 0
    t0 = time.time()
    done = True

    # --- loop
    while True:
        _time = time.time()

        # timeout
        elapsed_time = _time - t0
        if config.timeout > 0 and elapsed_time > config.timeout:
            break

        # max steps
        if config.max_steps > 0 and total_step > config.max_steps:
            break

        # ------------------------
        # episode end / init
        # ------------------------
        if done:
            episode_count += 1

            if config.max_episodes > 0 and episode_count >= config.max_episodes:
                break  # end

            # env reset
            episode_t0 = _time
            state, next_player_indices = env.reset()
            done = False
            step = 0
            episode_rewards = np.zeros(env.player_num)

            # players reset
            players_status = ["INIT" for _ in range(env.player_num)]
            players_step_reward = np.zeros(env.player_num)
            worker_info_list = [None for _ in range(env.player_num)]
            worker_indices = [i for i in range(env.player_num)]

            if config.shuffle_player:
                random.shuffle(worker_indices)

            # callback
            _params["episode_count"] = episode_count
            _params["state"] = state
            _params["next_player_indices"] = next_player_indices
            _params["worker_indices"] = worker_indices
            [c.on_episode_begin(**_params) for c in callbacks]

        # ------------------------
        # step
        # ------------------------
        invalid_actions_list = [env.fetch_invalid_actions(idx) for idx in next_player_indices]

        # --- rl init
        for i, idx in enumerate(next_player_indices):
            if players_status[idx] == "INIT":
                worker_idx = worker_indices[idx]
                workers[worker_idx].on_reset(state, invalid_actions_list[i], env)
                players_status[idx] = "RUNNING"

        # callback
        _params["step"] = step
        [c.on_step_begin(**_params) for c in callbacks]

        # --- rl action
        actions = []
        for i, idx in enumerate(next_player_indices):
            worker_idx = worker_indices[idx]
            action = workers[worker_idx].policy(state, invalid_actions_list[i], env)
            actions.append(action)

        # --- env step
        # skip frame の間は同じアクションを繰り返す
        rewards = np.zeros(env.player_num)
        for j in range(config.skip_frames + 1):
            state, step_rewards, done, next_player_indices, env_info = env.step(actions)
            rewards += np.asarray(step_rewards)
            if done:
                break

            # callback
            if j < config.skip_frames:
                _params["state"] = state
                _params["step_rewards"] = step_rewards
                _params["done"] = done
                _params["env_info"] = env_info
                [c.on_skip_step(**_params) for c in callbacks]

        step += 1
        total_step += 1

        # update reward
        episode_rewards += rewards
        players_step_reward += rewards

        # --- episode end
        if step >= env.max_episode_steps:
            done = True
        if step >= config.max_episode_steps:
            done = True
        if _time - episode_t0 > config.episode_timeout:
            done = True

        # --- rl after step
        if done:
            # 終了の場合は全playerを実行
            next_player_indices = [i for i in range(env.player_num)]
        for idx in next_player_indices:
            if players_status[idx] != "RUNNING":
                continue
            invalid_actions = env.fetch_invalid_actions(idx)
            worker_idx = worker_indices[idx]
            worker_info_list[worker_idx] = workers[worker_idx].on_step(
                state, players_step_reward[idx], done, invalid_actions, env
            )
            players_step_reward[idx] = 0

        # --- trainer
        if config.training and trainer is not None:
            _t0 = time.time()
            train_info = trainer.train()
            train_time = time.time() - _t0
        else:
            train_info = None
            train_time = 0

        # callback
        _params["step"] = step
        _params["step_time"] = time.time() - _time
        _params["state"] = state
        _params["actions"] = actions
        _params["rewards"] = rewards
        _params["done"] = done
        _params["env_info"] = env_info
        _params["worker_info_list"] = worker_info_list
        _params["train_info"] = train_info
        _params["train_time"] = train_time
        [c.on_step_end(**_params) for c in callbacks]

        if done:
            episode_rewards_list.append(episode_rewards)

            # callback
            _params["step"] = step
            _params["episode_time"] = _time - episode_t0
            _params["episode_count"] = episode_count
            _params["episode_rewards"] = episode_rewards
            [c.on_episode_end(**_params) for c in callbacks]

        # callback end
        if True in [c.intermediate_stop(**_params) for c in callbacks]:
            break

    # callback
    _params["episode_count"] = episode_count
    [c.on_episodes_end(**_params) for c in callbacks]

    if env.player_num == 1:
        return [r[0] for r in episode_rewards_list], parameter, remote_memory
    else:
        return episode_rewards_list, parameter, remote_memory


if __name__ == "__main__":
    pass

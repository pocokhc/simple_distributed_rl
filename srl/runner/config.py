import logging
import pickle
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

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

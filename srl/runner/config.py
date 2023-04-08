import enum
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

import srl
import srl.rl.dummy
import srl.rl.human  # reservation
import srl.rl.random_play  # reservation
from srl.base.define import PlayRenderMode
from srl.base.env.base import EnvRun
from srl.base.env.config import EnvConfig
from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory, RLTrainer
from srl.base.rl.registration import (make_parameter, make_remote_memory,
                                      make_trainer, make_worker,
                                      make_worker_rulebase)
from srl.base.rl.worker import WorkerRun
from srl.runner.callback import Callback
from srl.utils.common import is_package_imported

logger = logging.getLogger(__name__)


@dataclass
class Config:

    env_config: Union[str, EnvConfig]
    rl_config: RLConfig

    # multi player option
    players: List[Union[None, str, RLConfig]] = field(default_factory=list)

    # mp options
    actor_num: int = 1
    trainer_parameter_send_interval_by_train_count: int = 100
    actor_parameter_sync_interval_by_step: int = 100
    allocate_main: str = ""
    allocate_trainer: str = "/GPU:0"
    allocate_actor: Union[List[str], str] = "/CPU:0"

    # tensorflow options
    tf_enable_gpu: bool = True
    tf_enable_memory_growth: bool = True
    tf_on_init_function: Optional[Callable[["Config"], None]] = None  # tfの初期化を追加で設定できます

    def __post_init__(self):
        # stop config
        self.max_episodes: int = -1
        self.timeout: int = -1
        self.max_steps: int = -1
        self.max_train_count: int = -1
        # play config
        self.shuffle_player: bool = False
        self.disable_trainer: bool = False
        self.seed: Optional[int] = None
        self.render_mode: PlayRenderMode = PlayRenderMode.none
        self.render_kwargs: dict = {}
        self.enable_profiling: bool = True
        # callbacks
        self.callbacks: List[Callback] = []

        # play info
        self.training: bool = False
        self.distributed: bool = False
        self.enable_ps: bool = False
        self.enable_nvidia: bool = False
        self.run_name: str = "main"
        self.run_actor_id: int = 0

        if self.rl_config is None:
            self.rl_config = srl.rl.dummy.Config()
        if isinstance(self.env_config, str):
            self.env_config = EnvConfig(self.env_config)

        self.rl_name = self.rl_config.getName()
        self.env = None

        self._is_init_tensorflow: bool = False
        self._default_CUDA_VISIBLE_DEVICES: Optional[str] = None

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
        assert self.actor_num > 0
        self.rl_config.assert_params()

    def _set_env(self):
        if self.env is None:
            self.env = srl.make_env(self.env_config)
            self.rl_config.reset_config(self.env)

    def make_env(self) -> EnvRun:
        self.init_tensorflow()
        self._set_env()
        self.env.init()
        return self.env

    def make_parameter(self, is_load: bool = True) -> RLParameter:
        self.init_tensorflow()
        self._set_env()
        return make_parameter(self.rl_config, env=self.env, is_load=is_load)

    def make_remote_memory(self, is_load: bool = True) -> RLRemoteMemory:
        self.init_tensorflow()
        self._set_env()
        return make_remote_memory(self.rl_config, env=self.env, is_load=is_load)

    def make_trainer(self, parameter: RLParameter, remote_memory: RLRemoteMemory) -> RLTrainer:
        self.init_tensorflow()
        self._set_env()
        return make_trainer(self.rl_config, parameter, remote_memory, env=self.env)

    def make_worker(
        self,
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
        actor_id: int = 0,
    ) -> WorkerRun:
        self.init_tensorflow()
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
        self.init_tensorflow()
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
            assert False, f"not registered: {player_obj}"

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
        conf = {}
        for k, v in self.__dict__.items():
            if v is None or type(v) in [int, float, bool, str]:
                conf[k] = v
            elif type(v) is list:
                conf[k] = [str(n) for n in v]
            elif type(v) is dict:
                conf[k] = v.copy()
            elif issubclass(type(v), enum.Enum):
                conf[k] = v.name

        conf["rl_config"] = {}
        for k, v in self.rl_config.__dict__.items():
            if v is None or type(v) in [int, float, bool, str]:
                conf["rl_config"][k] = v
            elif type(v) is list:
                conf["rl_config"][k] = [str(n) for n in v]
            elif type(v) is dict:
                conf["rl_config"] = v.copy()
            elif issubclass(type(v), enum.Enum):
                conf["rl_config"][k] = v.name

        conf["env_config"] = {}
        for k, v in self.env_config.__dict__.items():
            if v is None or type(v) in [int, float, bool, str]:
                conf["env_config"][k] = v
            elif type(v) is list:
                conf["env_config"][k] = [str(n) for n in v]
            elif type(v) is dict:
                conf["env_config"] = v.copy()
            elif issubclass(type(v), enum.Enum):
                conf["env_config"][k] = v.name

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
            elif issubclass(type(v), enum.Enum):
                setattr(config, k, v)

        # list parameter
        config.players = []
        for player in self.players:
            if player is None:
                config.players.append(None)
            else:
                config.players.append(pickle.loads(pickle.dumps(player)))

        # callback
        if callbacks_share:
            config.callbacks = self.callbacks
        else:
            config.callbacks = pickle.loads(pickle.dumps(self.callbacks))

        # env
        if env_share:
            config.env = self.env

        return config

    def get_allocate(self) -> str:
        if self.run_name in ["main", "eval"]:
            if self.allocate_main == "" and self.distributed:
                return "/CPU:0"
            return self.allocate_main
        elif self.run_name == "trainer":
            return self.allocate_trainer
        elif "actor" in self.run_name:
            if isinstance(self.allocate_actor, str):
                return self.allocate_actor
            else:
                return self.allocate_actor[self.run_actor_id]
        raise ValueError()

    def init_tensorflow(self, rerun: bool = False):
        """
        tensorflow の初期化はmodel作成前にしかできない
        """
        if not rerun and self._is_init_tensorflow:
            return
        if not is_package_imported("tensorflow"):
            self._is_init_tensorflow = True
            return
        import tensorflow as tf

        if self._default_CUDA_VISIBLE_DEVICES is None:
            self._default_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "None")
            logger.info(f"[{self.run_name}] default CUDA_VISIBLE_DEVICES={self._default_CUDA_VISIBLE_DEVICES}")

        if not self.tf_enable_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            logger.info(f"[{self.run_name}] set CUDA_VISIBLE_DEVICES=-1")
        else:
            allocate = self.get_allocate()
            logger.info(f"[{self.run_name}] allocate={allocate}")

            if "CPU" in allocate:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                logger.info(f"[{self.run_name}] set CUDA_VISIBLE_DEVICES=-1")
            else:
                #  set default allocate
                if self._default_CUDA_VISIBLE_DEVICES is None or self._default_CUDA_VISIBLE_DEVICES == "None":
                    if "CUDA_VISIBLE_DEVICES" in os.environ:
                        del os.environ["CUDA_VISIBLE_DEVICES"]
                        logger.info(f"[{self.run_name}] del CUDA_VISIBLE_DEVICES")
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = self._default_CUDA_VISIBLE_DEVICES
                    logger.info(f"[{self.run_name}] set CUDA_VISIBLE_DEVICES={self._default_CUDA_VISIBLE_DEVICES}")

                if self.tf_enable_memory_growth:
                    try:
                        for device in tf.config.list_physical_devices("GPU"):
                            logger.info(f"[{self.run_name}] set_memory_growth({device.name}, True)")
                            tf.config.experimental.set_memory_growth(device, True)
                    except Exception:
                        print(f"[{self.run_name}] 'set_memory_growth' failed. Also consider 'tf_enable_memory_growth=False'.")
                        raise

        if self.tf_on_init_function is not None:
            self.tf_on_init_function(self)

        self._is_init_tensorflow = True
        logger.info(f"[{self.run_name}] Initialized tensorflow.")

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

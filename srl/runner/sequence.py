import logging
import pickle
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import gym
from srl import rl
from srl.base.rl import RLTrainer, RLWorker
from srl.base.rl.config import RLConfig
from srl.base.rl.env_for_rl import EnvForRL
from srl.base.rl.rl import RLParameter, RLRemoteMemory
from srl.runner.callbacks import Callback

logger = logging.getLogger(__name__)


@dataclass
class Config:

    env_name: str
    rl_config: RLConfig

    # episode option
    max_episode_steps: int = 10_000
    episode_timeout: int = 60 * 10  # s

    # play option
    max_steps: int = -1
    max_episodes: int = -1
    timeout: int = -1
    training: bool = False

    # other
    callbacks: List[Callback] = field(default_factory=list)

    def __post_init__(self):
        if self.rl_config is not None:
            self.rl_name = self.rl_config.getName()
        self.is_init_rl_config = False
        self.parameter_path = ""
        self.memory_path = ""
        self.worker_id = 0
        self.trainer_disable = False

    def set_play_config(
        self,
        max_steps: int = -1,
        max_episodes: int = -1,
        timeout: int = -1,
        training: bool = False,
        parameter_path: str = "",
        callbacks: List[Callback] = None,
    ):
        if callbacks is None:
            callbacks = []
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.timeout = timeout
        self.training = training
        self.parameter_path = parameter_path
        self.callbacks = callbacks

    # ----------------------------------------------

    def assert_params(self):
        self.rl_config.assert_params()

    def init_rl_config(self) -> None:
        if not self.is_init_rl_config:
            self.create_env()

    def create_env(self, **kwargs) -> EnvForRL:
        env = EnvForRL(gym.make(self.env_name), self.rl_config, **kwargs)
        self.is_init_rl_config = True
        return env

    def create_parameter(self) -> RLParameter:
        self.init_rl_config()
        module = rl.make(self.rl_name)
        parameter = module.Parameter(self.rl_config)
        parameter.load(self.parameter_path)
        return parameter

    def create_remote_memory(self) -> RLRemoteMemory:
        self.init_rl_config()
        module = rl.make(self.rl_name)
        memory = module.RemoteMemory(self.rl_config)
        memory.load(self.memory_path)
        return memory

    def create_trainer(self, parameter: RLParameter, memory: RLRemoteMemory) -> RLTrainer:
        self.init_rl_config()
        module = rl.make(self.rl_name)
        trainer = module.Trainer(self.rl_config, parameter, memory)
        return trainer

    def create_worker(self, parameter: Optional[RLParameter], memory: Optional[RLRemoteMemory]) -> RLWorker:
        self.init_rl_config()
        module = rl.make(self.rl_name)
        worker = module.Worker(self.rl_config, parameter, memory, self.worker_id)
        worker.set_training(self.training)
        return worker

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
        config.rl_config = pickle.loads(pickle.dumps(self.rl_config))
        return config


def play(
    config: Config,
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
    env=None,
) -> Tuple[List[float], RLParameter, RLRemoteMemory]:
    if env is None:
        env = config.create_env()
    if parameter is None:
        parameter = config.create_parameter()
    if remote_memory is None:
        remote_memory = config.create_remote_memory()
    if config.training and not config.trainer_disable:
        trainer = config.create_trainer(parameter, remote_memory)
    else:
        trainer = None
    worker = config.create_worker(parameter, remote_memory)
    config.assert_params()

    # callback
    callbacks = config.callbacks
    _params = {
        "config": config,
        "env": env,
        "parameter": parameter,
        "remote_memory": remote_memory,
        "trainer": trainer,
        "worker": worker,
    }
    [c.on_episodes_begin(_params) for c in callbacks]

    # --- rewards
    episode_rewards = []

    # --- init
    episode_count = -1
    total_step = 0
    t0 = time.time()
    done = True

    state = None
    step = 0
    episode_reward = 0
    valid_actions = None
    episode_t0 = 0

    # --- loop
    while True:
        _time = time.time()

        # timeout
        elapsed_time = _time - t0
        if config.timeout > 0 and elapsed_time > config.timeout:
            break

        # step end
        if config.max_steps > 0 and total_step > config.max_steps:
            break

        # ------------------------
        # episode end
        # ------------------------
        if done:
            episode_count += 1

            if config.max_episodes > 0 and episode_count >= config.max_episodes:
                break  # end

            # env reset
            state = env.reset()
            done = False
            step = 0
            episode_reward = 0
            valid_actions = env.fetch_valid_actions()
            episode_t0 = _time

            # rl reset
            worker.on_reset(state, valid_actions, env)

            # callback
            _params = {
                "config": config,
                "env": env,
                "parameter": parameter,
                "remote_memory": remote_memory,
                "trainer": trainer,
                "worker": worker,
                "episode_count": episode_count,
                "state": state,
                "valid_actions": valid_actions,
            }
            [c.on_episode_begin(_params) for c in callbacks]

        # ------------------------
        # step
        # ------------------------

        # callback
        _params = {
            "config": config,
            "env": env,
            "parameter": parameter,
            "remote_memory": remote_memory,
            "trainer": trainer,
            "worker": worker,
            "episode_count": episode_count,
            "step": step,
        }
        [c.on_step_begin(_params) for c in callbacks]

        # action
        env_action, worker_action = worker.policy(state, valid_actions, env)
        if valid_actions is not None:
            assert env_action in valid_actions

        # env step
        next_state, reward, done, env_info = env.step(env_action)
        episode_reward += reward
        step += 1
        next_valid_actions = env.fetch_valid_actions()
        total_step += 1

        # --- episode end
        if step >= env.max_episode_steps:
            done = True
        if step >= config.max_episode_steps:
            done = True
        if _time - episode_t0 > config.episode_timeout:
            done = True

        # --- rl step
        work_info = worker.on_step(
            state,
            worker_action,
            next_state,
            reward,
            done,
            valid_actions,
            next_valid_actions,
            env,
        )
        if config.training and trainer is not None:
            _t0 = time.time()
            train_info = trainer.train()
            train_time = time.time() - _t0
        else:
            train_info = None
            train_time = 0

        # callback
        _params = {
            "config": config,
            "env": env,
            "parameter": parameter,
            "remote_memory": remote_memory,
            "trainer": trainer,
            "worker": worker,
            "episode_count": episode_count,
            "step": step,
            "step_time": time.time() - _time,
            "state": next_state,
            "action": env_action,
            "valid_actions": next_valid_actions,
            "reward": reward,
            "done": done,
            "env_info": env_info,
            "work_info": work_info,
            "train_info": train_info,
            "train_time": train_time,
        }
        [c.on_step_end(_params) for c in callbacks]

        # callback end
        if True in [c.intermediate_stop(_params) for c in callbacks]:
            break

        state = next_state
        valid_actions = next_valid_actions

        if done:
            episode_rewards.append(episode_reward)

            # callback
            _params = {
                "config": config,
                "env": env,
                "parameter": parameter,
                "remote_memory": remote_memory,
                "trainer": trainer,
                "worker": worker,
                "episode_count": episode_count,
                "episode_time": _time - episode_t0,
                "step": step,
                "reward": episode_reward,
            }
            [c.on_episode_end(_params) for c in callbacks]

    # callback
    _params = {
        "config": config,
        "env": env,
        "parameter": parameter,
        "remote_memory": remote_memory,
        "trainer": trainer,
        "worker": worker,
        "episode_count": episode_count,
    }
    [c.on_episodes_end(_params) for c in callbacks]

    return episode_rewards, parameter, remote_memory


if __name__ == "__main__":
    pass

import logging
import pickle
import time
from dataclasses import dataclass, field
from typing import Optional, cast

import srl.rl.memory.registory
from srl import rl
from srl.base.rl import RLTrainer, RLWorker
from srl.base.rl.config import RLConfig
from srl.base.rl.env_for_rl import EnvForRL, create_env_for_rl
from srl.base.rl.memory import Memory, MemoryConfig
from srl.base.rl.rl import RLParameter
from srl.runner.callbacks import Callback

logger = logging.getLogger(__name__)


@dataclass
class Config:

    env_name: str
    rl_config: RLConfig
    memory_config: MemoryConfig

    # episode option
    max_episode_steps: int = 10_000
    episode_timeout: int = 60 * 10  # s

    # play option
    max_steps: int = -1
    max_episodes: int = -1
    timeout: int = -1
    training: bool = False

    # other
    callbacks: list[Callback] = field(default_factory=list)

    def __post_init__(self):
        if self.rl_config is not None:
            self.rl_name = self.rl_config.getName()
        if self.memory_config is not None:
            self.memory_name = self.memory_config.getName()
        self.is_init_rl_config = False
        self.parameter_path = ""
        self.worker_id = 0
        self.trainer_disable = False

    def set_play_config(
        self,
        max_steps: int = -1,
        max_episodes: int = -1,
        timeout: int = -1,
        training: bool = False,
        parameter_path: str = "",
        callbacks: list[Callback] = None,
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
        self.create_env()

    def create_env(self) -> EnvForRL:
        env = create_env_for_rl(self.env_name, self.rl_config)
        self.is_init_rl_config = True
        return env

    def create_parameter(self) -> RLParameter:
        if not self.is_init_rl_config:
            self.init_rl_config()
        module = rl.make(self.rl_name)
        parameter = module.Parameter(self.rl_config)
        parameter.load(self.parameter_path)
        return parameter

    def create_memory(self) -> Memory:
        memory = srl.rl.memory.registory.make(self.memory_config)
        return memory

    def create_trainer(self, parameter: RLParameter) -> RLTrainer:
        if not self.is_init_rl_config:
            self.init_rl_config()
        module = rl.make(self.rl_name)
        trainer = module.Trainer(self.rl_config, parameter)
        return trainer

    def create_worker(self, parameter: RLParameter) -> RLWorker:
        if not self.is_init_rl_config:
            self.init_rl_config()
        module = rl.make(self.rl_name)
        worker = module.Worker(self.rl_config, parameter, self.worker_id)
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

        conf["memory_config"] = {}
        for k, v in self.memory_config.__dict__.items():
            if type(v) in [int, float, bool, str]:
                conf["memory_config"][k] = v

        return conf

    def copy(self):
        config = Config("", None, None)  # type: ignore
        for k, v in self.__dict__.items():
            if type(v) in [int, float, bool, str]:
                setattr(config, k, v)
        config.rl_config = pickle.loads(pickle.dumps(self.rl_config))
        config.memory_config = pickle.loads(pickle.dumps(self.memory_config))
        return config


def play(
    config: Config,
    parameter: Optional[RLParameter] = None,
    memory: Optional[Memory] = None,
    env=None,
) -> tuple[list[float], RLParameter, Memory]:
    if env is None:
        env = config.create_env()
    if parameter is None:
        parameter = config.create_parameter()
    if memory is None:
        memory = config.create_memory()
    if config.training and not config.trainer_disable:
        trainer = config.create_trainer(parameter)
    else:
        trainer = None
    worker = config.create_worker(parameter)
    config.assert_params()

    # callback
    callbacks = config.callbacks
    _params = {
        "config": config,
        "env": env,
        "parameter": parameter,
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
            worker.on_reset(state, valid_actions)

            # callback
            _params = {
                "config": config,
                "env": env,
                "parameter": parameter,
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
            "trainer": trainer,
            "worker": worker,
            "episode_count": episode_count,
            "step": step,
        }
        [c.on_step_begin(_params) for c in callbacks]

        # action
        env_action, worker_action = worker.policy(state, valid_actions)
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
        work_return = worker.on_step(state, worker_action, next_state, reward, done, valid_actions, next_valid_actions)
        if config.training:
            batch, priority, work_info = work_return
            priority = cast(float, priority)
            memory.add(batch, priority)
            if trainer is None:
                train_info = None
            else:
                train_info = trainer.train(memory)
        else:
            work_info = work_return
            priority = 0
            train_info = None

        # callback
        _params = {
            "config": config,
            "env": env,
            "parameter": parameter,
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
            "priority": priority,
            "env_info": env_info,
            "work_info": work_info,
            "train_info": train_info,
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
        "trainer": trainer,
        "worker": worker,
        "episode_count": episode_count,
    }
    [c.on_episodes_end(_params) for c in callbacks]

    return episode_rewards, parameter, memory


if __name__ == "__main__":
    pass

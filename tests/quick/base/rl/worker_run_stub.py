from typing import Any

import numpy as np

from srl.base.define import RLActionType, RLBaseTypes
from srl.base.env.base import EnvBase
from srl.base.rl.config import RLConfig
from srl.base.rl.worker import RLWorker
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.space import SpaceBase


class WorkerRunStubEnv(EnvBase):
    def __init__(
        self,
        action_space=DiscreteSpace(7),
        observation_space=DiscreteSpace(7),
        invalid_action=False,
    ):
        self._action_space = action_space
        self._observation_space = observation_space
        self.invalid_action = invalid_action

        self.s_states: list = [1] * 10
        self.s_reward = 0.0
        self.s_action = 0

    @property
    def action_space(self) -> SpaceBase:
        return self._action_space

    @property
    def observation_space(self) -> SpaceBase:
        return self._observation_space

    @property
    def player_num(self) -> int:
        return 1

    @property
    def max_episode_steps(self) -> int:
        return 5

    def reset(self, **kwargs):
        self.num_step = 1
        return self.s_states[self.num_step - 1]

    def step(self, action):
        self.s_action = action
        self.num_step += 1
        self.s_reward += 1
        done = self.num_step - 1 == len(self.s_states) - 1
        return self.s_states[self.num_step - 1], self.s_reward, done, False

    def get_invalid_actions(self, player_index: int = -1):
        if self.invalid_action:
            return [self.num_step]
        else:
            return []

    def backup(self, **kwargs) -> Any:
        return None

    def restore(self, state: Any, **kwargs) -> None:
        pass  # do nothing

    def render_terminal(self):
        print("a")

    def render_rgb_array(self, **kwargs) -> np.ndarray:
        return np.full((64, 32, 3), self.num_step)


class WorkerRunStubRLConfig(RLConfig):
    def __init__(self, name="Stub") -> None:
        super().__init__()
        self._name = name
        self._action_type = RLBaseTypes.DISCRETE
        self._observation_type = RLBaseTypes.ARRAY_DISCRETE
        self._use_render_image_state = False

    def get_name(self) -> str:
        return self._name

    def get_base_action_type(self) -> RLBaseTypes:
        return self._action_type

    def get_base_observation_type(self) -> RLBaseTypes:
        return self._observation_type

    def use_render_image_state(self) -> bool:
        return self._use_render_image_state


class WorkerRunStubRLWorker(RLWorker):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.on_reset_state = np.array(0)
        self.state = np.array(0)
        self.action = 0

    def on_reset(self, worker):
        self.on_reset_state = worker.state

    def policy(self, worker) -> RLActionType:
        self.state = worker.state
        return self.action

    def on_step(self, worker):
        self.state = worker.state

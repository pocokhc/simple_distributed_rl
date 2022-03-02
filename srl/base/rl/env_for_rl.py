import itertools
import logging
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union, cast

import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvObservationType, RLActionType, RLObservationType
from srl.base.env.env import EnvBase, GymEnvWrapper
from srl.base.rl.config import RLConfig

logger = logging.getLogger(__name__)


def create_env_for_rl(env_name: str, rl_config: RLConfig):
    env = EnvForRL(gym.make(env_name), rl_config)
    rl_config.set_config_by_env(env)
    return env


@dataclass
class EnvForRL:

    env: Union[gym.Env, EnvBase]
    config: RLConfig
    action_division_count: int = 5
    observation_division_count: int = 50
    prediction_by_simulation: bool = True

    # コンストラクタ
    def __post_init__(self):
        self._valid_actions = None  # cache

        # gym env
        if not issubclass(self.env.unwrapped.__class__, EnvBase):
            self.env = GymEnvWrapper(self.env)
        self.env = cast(EnvBase, self.env)

        # 変更前
        self.before_action_space = self.env.action_space
        self.before_observation_space = self.env.observation_space

        # アクションを変換
        if self.config.action_type == RLActionType.DISCRETE:
            self.after_action_space, self.action_change_type = self._action_discrete(self.env.action_space)
        elif self.config.action_type == RLActionType.CONTINUOUS:
            self.after_action_space, self.action_change_type = self._to_box(self.env.action_space)
        else:
            raise ValueError()

        # 状態はboxで統一
        self.after_observation_space, self.observation_change_type = self._to_box(self.env.observation_space)

        # 状態の離散化
        self._observation_discrete_diff = None
        if self.config.observation_type == RLObservationType.DISCRETE:
            self._observation_discrete()
            self.after_observation_type = EnvObservationType.DISCRETE
        else:
            self.after_observation_type = self.env.observation_type
            if self.after_observation_type == EnvObservationType.UNKOWN:
                self.after_observation_type = EnvObservationType.CONTINUOUS

        # 変更後
        logger.info(f"before_action          : {self._space_str(self.env.action_space)}")
        logger.info(f"before_observation     : {self._space_str(self.env.observation_space)}")
        logger.info(f"before_observation type: {self.env.observation_type}")
        logger.info(f"action_change      : {self.action_change_type}")
        logger.info(f"observation_change : {self.observation_change_type}")
        logger.info(f"after_action           : {self._space_str(self.after_action_space)}")
        logger.info(f"after_observation      : {self._space_str(self.after_observation_space)}")
        logger.info(f"after_observation type : {self.after_observation_type}")

    def _space_str(self, space):
        if isinstance(space, gym.spaces.Discrete):
            return f"{space.__class__.__name__} {space.n}"  # type: ignore
        if isinstance(space, gym.spaces.Tuple):
            return f"{space.__class__.__name__} {len(space)}"  # type: ignore
        return f"{space.__class__.__name__}{space.shape} ({space.low.flatten()[0]} - {space.high.flatten()[0]})"

    # アクションの離散化
    def _action_discrete(self, space):
        if isinstance(space, gym.spaces.Discrete):
            return space, ""

        if isinstance(space, gym.spaces.Tuple):
            self.action_tbl = list(itertools.product(*[[n for n in range(s.n)] for s in space.spaces]))  # type: ignore
            next_space = gym.spaces.Discrete(len(self.action_tbl))
            return next_space, "Tuple->Discrete"

        if isinstance(space, gym.spaces.Box):
            division_count = self.action_division_count

            shape = space.shape  # type: ignore
            low_flatten = space.low.flatten()  # type: ignore
            high_flatten = space.high.flatten()  # type: ignore

            act_list = []
            for i in range(len(low_flatten)):
                act = []
                for j in range(division_count):
                    low = low_flatten[i]
                    high = high_flatten[i]
                    diff = (high - low) / (division_count - 1)

                    a = low + diff * j
                    act.append(a)
                act_list.append(act)

            act_list = list(itertools.product(*act_list))
            self.action_tbl = np.reshape(act_list, (-1,) + shape).tolist()

            next_space = gym.spaces.Discrete(len(self.action_tbl))
            return next_space, "Box->Discrete"

        raise ValueError(f"{space.__class__.__name__}")

    def _action_encode(self, action) -> Any:
        if self.action_change_type == "":
            return action
        if self.action_change_type == "Discrete->Box":
            return np.round(action[0])
        if self.action_change_type == "Tuple->Box":
            return action
        if self.action_change_type == "Tuple->Discrete":
            return self.action_tbl[action]
        if self.action_change_type == "Box->Discrete":
            return [self.action_tbl[action]]

        raise ValueError()

    def _valid_actions_encode(self):
        if self.action_change_type == "":
            return None
        if self.action_change_type == "Discrete->Box":
            return None
        if self.action_change_type == "Tuple->Box":
            return None
        if self.action_change_type == "Box->Discrete":
            return [a for a in range(len(self.action_tbl))]

        raise ValueError()

    # Box に変換
    def _to_box(self, space):
        if isinstance(space, gym.spaces.Discrete):
            next_space = gym.spaces.Box(low=0, high=space.n - 1, shape=(1,))  # type: ignore
            return next_space, "Discrete->Box"

        if isinstance(space, gym.spaces.Tuple):
            low = []
            high = []
            shape_num = 0
            for s in space.spaces:  # type: ignore
                if isinstance(s, gym.spaces.Discrete):
                    low.append(0)
                    high.append(s.n)  # type: ignore
                    shape_num += 1
                    continue

                if isinstance(s, gym.spaces.Box) and len(s.shape) == 1:  # type: ignore
                    for _ in range(s.shape[0]):  # type: ignore
                        low.append(s.low)  # type: ignore
                        high.append(s.high)  # type: ignore
                        shape_num += 1
                    continue

                raise ValueError(f"{s.__class__.__name__}")
            next_space = gym.spaces.Box(
                low=np.array(low), high=np.array(high), shape=(len(space.spaces),)  # type: ignore
            )
            return next_space, "Tuple->Box"

        if isinstance(space, gym.spaces.Box):
            return space, ""  # no change

        raise ValueError(f"{space.__class__.__name__}")

    # 状態の離散化
    def _observation_discrete(self):
        space = self.before_observation_space
        self._observation_discrete_diff = None

        # Discrete は離散
        if isinstance(space, gym.spaces.Discrete):
            logger.info("observation type: discrete")
            return

        if isinstance(space, gym.spaces.Tuple):
            # 全部Discreteなら離散
            is_discrete = True
            for s in space.spaces:  # type: ignore
                if not isinstance(s, gym.spaces.Discrete):
                    is_discrete = False
                    break

            if is_discrete:
                logger.info("observation type: discrete")
                return

        # 実際の値を取得して、小数がある場合は離散化する
        if self.prediction_by_simulation:
            self.env = cast(EnvBase, self.env)
            is_discrete = True
            done = True
            for _ in range(100):
                if done:
                    self.env.reset()
                va = self.env.fetch_valid_actions()
                if va is None:
                    action = self.before_action_space.sample()
                else:
                    action = random.choice(va)
                state, reward, done, _ = self.env.step(action)
                if "int" not in str(np.asarray(state).dtype):
                    is_discrete = False
            if is_discrete:
                logger.info("observation type: discrete")
                return

        # --- 離散化
        division_count = self.observation_division_count
        low = self.after_observation_space.low  # type: ignore
        high = self.after_observation_space.high  # type: ignore
        self._observation_discrete_diff = (high - low) / division_count

        logger.info(f"observation type: continuous(division {division_count})")

    def _observation_encode_discrete(self, state):
        if self._observation_discrete_diff is None:
            return state
        next_state = (state - self.after_observation_space.low) / self._observation_discrete_diff  # type: ignore
        next_state = np.int64(next_state)
        return next_state

    def _observation_encode(self, state):
        if self.observation_change_type == "":
            pass
        elif self.observation_change_type == "Discrete->Box":
            state = [state]
        elif self.observation_change_type == "Tuple->Box":
            pass
        else:
            raise ValueError()
        state = np.asarray(state)
        # assert state.shape == self.after_observation_space.shape
        state = state.reshape(self.after_observation_space.shape)  # type: ignore
        state = self._observation_encode_discrete(state)
        return state

    # ------------------------------------------
    # ABC method
    # ------------------------------------------

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.after_action_space

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self.after_observation_space

    @property
    def observation_type(self) -> EnvObservationType:
        return self.after_observation_type

    @property
    def max_episode_steps(self) -> int:
        return self.env.max_episode_steps  # type: ignore

    def close(self) -> None:
        self.env.close()

    def reset(self) -> Any:
        state = self.env.reset()
        state = self._observation_encode(state)
        self._valid_actions = None
        return state

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        action = self._action_encode(action)
        state, reward, done, info = self.env.step(action)
        state = self._observation_encode(state)
        self._valid_actions = None
        return state, float(reward), bool(done), info

    def fetch_valid_actions(self) -> Optional[List[int]]:
        if self.config.action_type == RLActionType.CONTINUOUS:
            return None

        if self._valid_actions is None:
            self._valid_actions = self.env.fetch_valid_actions()  # type: ignore
            if self._valid_actions is None:
                self._valid_actions = self._valid_actions_encode()
        return self._valid_actions

    def render(self, mode: str = "human") -> Any:
        return self.env.render(mode)

    def backup(self) -> Any:
        return self.env.backup()  # type: ignore

    def restore(self, state: Any) -> None:
        return self.env.restore(state)  # type: ignore


if __name__ == "__main__":
    pass

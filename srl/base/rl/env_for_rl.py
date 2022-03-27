import itertools
import logging
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union, cast

import cv2
import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvObservationType, RLActionType, RLObservationType
from srl.base.env.env import EnvBase, GymEnvWrapper
from srl.base.rl.config import RLConfig

logger = logging.getLogger(__name__)


@dataclass
class EnvForRL(EnvBase):

    env: Union[gym.Env, EnvBase]
    config: RLConfig
    override_env_observation_type: EnvObservationType = EnvObservationType.UNKOWN

    action_division_num: int = 5
    observation_division_num: int = 50
    prediction_by_simulation: bool = True
    enable_image_gray: bool = True
    image_resize: Optional[Tuple[int, int]] = None

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

        # 画像の場合グレー化
        if self.enable_image_gray:
            if self.env.observation_type == EnvObservationType.GRAY_3ch:
                self.after_observation_type = EnvObservationType.GRAY_2ch
            elif self.env.observation_type == EnvObservationType.COLOR:
                self.after_observation_type = EnvObservationType.GRAY_2ch

        # configの上書き
        if self.override_env_observation_type != EnvObservationType.UNKOWN:
            self.after_observation_type = self.override_env_observation_type

        # 変更後
        logger.info(f"before_action          : {self._space_str(self.env.action_space)}")
        logger.info(f"before_observation     : {self._space_str(self.env.observation_space)}")
        logger.info(f"before_observation type: {self.env.observation_type}")
        logger.info(f"action_change      : {self.action_change_type}")
        logger.info(f"observation_change : {self.observation_change_type}")
        logger.info(f"after_action           : {self._space_str(self.after_action_space)}")
        logger.info(f"after_observation      : {self._space_str(self.after_observation_space)}")
        logger.info(f"after_observation type : {self.after_observation_type}")

        # RLConfig側を設定する
        self.config.set_config_by_env(self)

    def _space_str(self, space):
        if isinstance(space, gym.spaces.Discrete):
            return f"{space.__class__.__name__} {space.n}"
        if isinstance(space, gym.spaces.Tuple):
            return f"{space.__class__.__name__} {len(space)}"
        return f"{space.__class__.__name__}{space.shape} ({space.low.flatten()[0]} - {space.high.flatten()[0]})"

    # アクションの離散化
    def _action_discrete(self, space):
        if isinstance(space, gym.spaces.Discrete):
            return space, ""

        if isinstance(space, gym.spaces.Tuple):
            self.action_tbl = list(itertools.product(*[[n for n in range(s.n)] for s in space.spaces]))
            next_space = gym.spaces.Discrete(len(self.action_tbl))
            return next_space, "Tuple->Discrete"

        if isinstance(space, gym.spaces.Box):
            division_num = self.action_division_num

            shape = space.shape
            low_flatten = space.low.flatten()
            high_flatten = space.high.flatten()

            act_list = []
            for i in range(len(low_flatten)):
                act = []
                for j in range(division_num):
                    low = low_flatten[i]
                    high = high_flatten[i]
                    diff = (high - low) / (division_num - 1)

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
            next_space = gym.spaces.Box(low=0, high=space.n - 1, shape=(1,))
            return next_space, "Discrete->Box"

        if isinstance(space, gym.spaces.Tuple):
            low = []
            high = []
            shape_num = 0
            for s in space.spaces:
                if isinstance(s, gym.spaces.Discrete):
                    low.append(0)
                    high.append(s.n)
                    shape_num += 1
                    continue

                if isinstance(s, gym.spaces.Box) and len(s.shape) == 1:
                    for _ in range(s.shape[0]):
                        low.append(s.low)
                        high.append(s.high)
                        shape_num += 1
                    continue

                raise ValueError(f"{s.__class__.__name__}")
            next_space = gym.spaces.Box(low=np.array(low), high=np.array(high), shape=(len(space.spaces),))
            return next_space, "Tuple->Box"

        if isinstance(space, gym.spaces.Box):
            return space, ""  # no change

        raise ValueError(f"{space.__class__.__name__}")

    # 状態の離散化
    def _observation_discrete(self):
        self._observation_discrete_diff = None
        space = self.before_observation_space

        if self.env.observation_type not in [
            EnvObservationType.UNKOWN,
            EnvObservationType.CONTINUOUS,
        ]:
            logger.debug("observation type: discrete")
            return

        # Discrete は離散
        if isinstance(space, gym.spaces.Discrete):
            logger.debug("observation type: discrete")
            return

        if isinstance(space, gym.spaces.Tuple):
            # 全部Discreteなら離散
            is_discrete = True
            for s in space.spaces:
                if not isinstance(s, gym.spaces.Discrete):
                    is_discrete = False
                    break

            if is_discrete:
                logger.debug("observation type: discrete")
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
                logger.debug("observation type: discrete")
                return

        # --- 離散化
        division_num = self.observation_division_num
        low = self.after_observation_space.low
        high = self.after_observation_space.high
        self._observation_discrete_diff = (high - low) / division_num

        logger.debug(f"observation type: continuous(division {division_num})")

    def _observation_encode_discrete(self, state):
        if self._observation_discrete_diff is None:
            return state
        next_state = (state - self.after_observation_space.low) / self._observation_discrete_diff
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

        # 画像の場合グレー化
        self.env = cast(EnvBase, self.env)
        if self.enable_image_gray:
            if self.env.observation_type == EnvObservationType.GRAY_3ch:
                # (w,h,1) -> (w,h)
                state = np.squeeze(state, -1)
            elif self.env.observation_type == EnvObservationType.COLOR:
                # (w,h,ch) -> (w,h)
                state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)

        # 画像のresize
        if self.image_resize is not None:
            if self.env.observation_type in [
                EnvObservationType.GRAY_2ch,
                EnvObservationType.GRAY_3ch,
                EnvObservationType.COLOR,
            ]:
                state = cv2.resize(state, self.image_resize[0], self.image_resize[1])

        # assert state.shape == self.after_observation_space.shape
        state = state.reshape(self.after_observation_space.shape)
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
        return self.env.max_episode_steps

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
            self._valid_actions = self.env.fetch_valid_actions()
            if self._valid_actions is None:
                self._valid_actions = self._valid_actions_encode()
        return self._valid_actions

    def render(self, mode: str = "human") -> Any:
        return self.env.render(mode)

    def action_to_str(self, action: Any) -> str:
        return self.env.action_to_str(action)

    def backup(self) -> Any:
        return self.env.backup()

    def restore(self, state: Any) -> None:
        return self.env.restore(state)


if __name__ == "__main__":
    pass

import copy
import pickle
from abc import abstractmethod
from typing import Any, List, Tuple, Union, cast

import kaggle_environments

from srl.base.define import EnvActionType, EnvObservationType
from srl.base.env.base import EnvBase
from srl.base.env.env_run import EnvRun
from srl.base.rl.algorithms.env_worker import EnvWorker


class KaggleWrapper(EnvBase):
    def __init__(self, name):
        self.__name = name
        self.env = kaggle_environments.make(name, debug=False)
        self.__player_index: int = 0
        self.__player_actions: List[Union[None, EnvActionType]] = []
        self.__rewards: List[float] = []
        self.__config = self.env.configuration
        self.__state = None
        self.__info = {}

    @property
    def name(self) -> int:
        return self.__name

    @property
    def config(self):
        return self.__config

    @property
    def obs(self) -> dict:
        return self.__obs

    @property
    def next_player_index(self) -> int:
        return self.__player_index

    def reset(self) -> Tuple[EnvObservationType, dict]:
        obs = self.env.reset(self.player_num)
        self.__set_rewards(obs)
        self.__set_player(obs)

        self.__obs = self.__create_player_state(obs)
        is_start_episode, state, player_index, info = self.encode_obs(self.__obs, self.config)
        self.__state = state
        self.__info = info

        return self.__state, self.__info

    def __set_rewards(self, obs):
        self.__rewards = [0.0 if o.reward is None else float(o.reward) for o in obs]

    def __set_player(self, obs):
        # ACTIVEのユーザのみ予想する
        self.__player_actions = []
        for i in range(self.player_num):
            if obs[i]["status"] == "ACTIVE":
                self.__player_actions.append(None)
            else:
                self.__player_actions.append(0)
        self.__player_index = 0
        self.__search_next_player()

    def __search_next_player(self):
        for i in range(self.__player_index, self.player_num):
            if self.__player_actions[i] is None:
                self.__player_index = i

                return
        self.__player_index = -1

    def __create_player_state(self, obs):
        # core.py __get_shared_state
        _obs = copy.deepcopy(obs[0]["observation"])
        _obs.update(obs[self.__player_index]["observation"])
        return _obs

    def step(self, action: EnvActionType) -> Tuple[EnvObservationType, List[float], bool, dict]:
        self.__player_actions[self.__player_index] = action

        self.__search_next_player()
        if self.__player_index == -1:
            # 全プレイヤーがアクションを選択した
            actions = [self.decode_action(a) for a in self.__player_actions]
            obs = self.env.step(actions)
            self.__set_rewards(obs)
            self.__set_player(obs)

            self.__obs = self.__create_player_state(obs)
            is_start_episode, state, player_index, info = self.encode_obs(self.__obs, self.config)
            self.__state = state
            self.__info = info

        assert self.__state is not None
        return (
            self.__state,
            self.__rewards,
            self.env.done,
            self.__info,
        )

    def direct_step(self, observation, configuration) -> Tuple[bool, EnvObservationType, int, dict]:
        return self.encode_obs(observation, configuration)

    @property
    def can_simulate_from_direct_step(self) -> bool:
        """kaggle_environmentsの中身を復元できないのでシミュレートできない"""
        return False

    @abstractmethod
    def encode_obs(self, observation, configuration) -> Tuple[bool, EnvObservationType, int, dict]:
        raise NotImplementedError()

    @abstractmethod
    def decode_action(self, action: EnvActionType) -> Any:
        raise NotImplementedError()

    def backup(self) -> Any:
        return [
            self.env.clone(),
            self.__player_index,
            self.__player_actions[:],
            self.__rewards[:],
            pickle.dumps(
                [
                    self.__state,
                    self.__info,
                ]
            ),
        ]

    def restore(self, data: Any) -> None:
        self.env = data[0].clone()
        self.__player_index = data[1]
        self.__player_actions = data[2][:]
        self.__rewards = data[3][:]
        d = pickle.loads(data[4])
        self.__state = d[0]
        self.__info = d[1]

    @property
    def unwrapped(self) -> object:
        return self.env

    def render_terminal(self, **kwargs) -> None:
        print(self.env.render(mode="ansi"))

    @property
    def render_interval(self) -> float:
        return 1000 / 1


class KaggleWorker(EnvWorker):
    def call_policy(self, env: EnvRun) -> Tuple[int, dict]:
        _env = cast(KaggleWrapper, env.env)
        return self.kaggle_policy(_env.obs, _env.config), {}

    @abstractmethod
    def kaggle_policy(self, observation, configuration):
        raise NotImplementedError()

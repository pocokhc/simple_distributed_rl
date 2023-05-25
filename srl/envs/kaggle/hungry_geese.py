import enum
import logging
from typing import List, Optional, Tuple

from kaggle_environments.envs.hungry_geese.hungry_geese import Action, greedy_agent

from srl.base.define import EnvObservationTypes
from srl.base.env import registration
from srl.base.env.kaggle_wrapper import KaggleWorker, KaggleWrapper
from srl.base.rl.worker import RuleBaseWorker
from srl.base.spaces import ArrayDiscreteSpace, DiscreteSpace

logger = logging.getLogger(__name__)


registration.register(
    id="hungry_geese",
    entry_point=__name__ + ":HungryGeese",
)


class FieldType(enum.Enum):
    NONE = 0
    GEESE1 = 1
    GEESE2 = 2
    GEESE3 = 3
    GEESE4 = 4
    GEESE5 = 5
    GEESE6 = 6
    GEESE7 = 7
    GEESE8 = 8
    FOOD = 9


class HungryGeese(KaggleWrapper):
    """
    configuration = {
        'actTimeout': 1,
        'columns': 11,
        'episodeSteps': 200,
        'hunger_rate': 40,
        'max_length': 99,
        'min_food': 2,
        'rows': 7,
        'runTimeout': 1200
    }
    observation = {
        'food': [46, 24],
        'geese': [[44], [71], [72], [36]],
        'index': 0,
        'remainingOverageTime': 60,
        'step': 0
    }
    """

    def __init__(self, player_num: int = 4):
        super().__init__("hungry_geese")
        self._player_num = player_num

        assert isinstance(self.env.configuration, dict)
        self.columns = self.env.configuration["columns"]
        self.rows = self.env.configuration["rows"]

    @property
    def action_space(self) -> DiscreteSpace:
        return DiscreteSpace(4)

    @property
    def observation_space(self) -> ArrayDiscreteSpace:
        return ArrayDiscreteSpace(11 * 7, low=0, high=9)

    @property
    def observation_type(self) -> EnvObservationTypes:
        return EnvObservationTypes.DISCRETE

    @property
    def max_episode_steps(self) -> int:
        return 200

    @property
    def player_num(self) -> int:
        return self._player_num

    def encode_obs(self, observation, configuration) -> Tuple[bool, List[int], int, dict]:
        step = observation.step
        player_index = observation.index

        is_start_episode = step == 0
        if is_start_episode:
            self.prev_action = None

        state = [0 for _ in range(11 * 7)]
        for n in observation.food:
            state[n] = FieldType.FOOD.value
        for index, geese in enumerate(observation.geese):
            for n in geese:
                state[n] = FieldType.GEESE1.value + index

        return is_start_episode, state, player_index, {}

    def decode_action(self, action):
        self.prev_action = action
        if action == 0:
            return Action.NORTH.name
        if action == 1:
            return Action.EAST.name
        if action == 2:
            return Action.SOUTH.name
        if action == 3:
            return Action.WEST.name

    def get_invalid_actions(self, player_index: int) -> List[int]:
        if self.prev_action == 0:
            return [2]
        if self.prev_action == 1:
            return [3]
        if self.prev_action == 2:
            return [0]
        if self.prev_action == 3:
            return [1]
        return []

    def make_worker(self, name: str, **kwargs) -> Optional[RuleBaseWorker]:
        if name == "greedy":
            return Greedy(**kwargs)
        return None


class Greedy(KaggleWorker):
    def kaggle_policy(self, observation, configuration):
        return greedy_agent(observation, configuration)


if __name__ == "__main__":
    from pprint import pprint

    from kaggle_environments import make

    env = make("hungry_geese", debug=True)
    pprint(env.configuration)

    obs = env.reset(4)
    pprint(obs[0]["observation"])

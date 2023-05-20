import numpy as np

import srl
import srl.rl.dummy
from srl.base.define import EnvObservation, EnvObservationType, RLObservationType
from srl.base.env.base import EnvRun
from srl.base.env.spaces.space import SpaceBase
from srl.base.rl.processor import Processor
from srl.runner import sequence


class TestProcessor:
    def run(self, processor: Processor, env_name: str) -> EnvRun:
        env_config = srl.EnvConfig(env_name)
        rl_config = srl.rl.dummy.Config()
        rl_config.processors = [processor]

        config = sequence.Config(env_config, rl_config)
        env = config.make_env()

        sequence.train(config, max_episodes=10)
        return env

    def change_observation_info(
        self,
        processor: Processor,
        env_name: str,
        after_type: EnvObservationType,
        after_space: SpaceBase,
        rl_observation_type: RLObservationType = RLObservationType.ANY,
    ):
        env = srl.make_env(env_name)

        new_space, new_type = processor.change_observation_info(
            env.observation_space,
            env.observation_type,
            rl_observation_type,
            env,
        )
        assert new_type == after_type
        assert new_space.__class__.__name__ == after_space.__class__.__name__
        assert new_space == after_space

    def observation_decode(
        self, processor: Processor, env_name: str, in_observation: EnvObservation, out_observation: EnvObservation
    ):
        env = srl.make_env(env_name)
        env.reset()

        new_observation = processor.process_observation(in_observation, env)
        np.testing.assert_array_equal(out_observation, new_observation)

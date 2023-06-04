import numpy as np

import srl
import srl.rl.dummy
from srl.base.define import EnvObservationType, EnvObservationTypes
from srl.base.env.env_run import EnvRun
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import Processor
from srl.base.spaces.space import SpaceBase
from srl.runner import facade_sequence


class TestProcessor:
    def run(self, processor: Processor, env_name: str) -> EnvRun:
        env_config = srl.EnvConfig(env_name)
        rl_config = srl.rl.dummy.Config()
        rl_config.processors = [processor]

        config = facade_sequence.Config(env_config, rl_config)
        env = config.make_env()

        facade_sequence.train(config, max_episodes=10)
        return env

    def preprocess_observation_space(
        self,
        processor: Processor,
        env_name: str,
        after_type: EnvObservationTypes,
        after_space: SpaceBase,
        rl_config: RLConfig = srl.rl.dummy.Config(),
    ):
        env = srl.make_env(env_name)

        new_space, new_type = processor.preprocess_observation_space(
            env.observation_space,
            env.observation_type,
            env,
            rl_config,
        )
        assert new_type == after_type
        assert new_space.__class__.__name__ == after_space.__class__.__name__
        assert new_space == after_space

    def preprocess_observation(
        self,
        processor: Processor,
        env_name: str,
        in_observation: EnvObservationType,
        out_observation: EnvObservationType,
        check_shape_only: bool = False,
    ):
        env = srl.make_env(env_name)
        env.reset()

        new_observation = processor.preprocess_observation(in_observation, env)
        if check_shape_only:
            assert out_observation.shape == new_observation.shape  # type: ignore
        else:
            np.testing.assert_array_equal(out_observation, new_observation)

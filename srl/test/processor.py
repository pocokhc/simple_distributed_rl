import numpy as np

import srl
from srl.base.define import EnvObservationType, EnvObservationTypes
from srl.base.env.env_run import EnvRun
from srl.base.rl.config import DummyRLConfig, RLConfig
from srl.base.rl.processor import Processor
from srl.base.spaces.space import SpaceBase


class TestProcessor:
    def run(self, processor: Processor, env_name: str) -> EnvRun:
        env_config = srl.EnvConfig(env_name)
        rl_config = DummyRLConfig()
        rl_config.processors = [processor]

        runner = srl.Runner(env_config, rl_config)
        env = runner.make_env()

        runner.train(max_episodes=10)
        return env

    def preprocess_observation_space(
        self,
        processor: Processor,
        env_name: str,
        after_type: EnvObservationTypes,
        after_space: SpaceBase,
        rl_config: RLConfig = DummyRLConfig(),
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

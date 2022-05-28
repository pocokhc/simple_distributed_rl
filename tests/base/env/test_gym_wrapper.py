import unittest
import warnings

from srl.base.define import EnvObservationType
from srl.base.env.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.env.spaces.box import BoxSpace
from srl.base.env.spaces.discrete import DiscreteSpace
from srl.test import TestEnv


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestEnv()
        warnings.simplefilter("ignore")

    def test_play_FrozenLake(self):
        # observation_space: Discrete
        # action_space     : Discrete
        env = self.tester.play_test("FrozenLake-v1")
        self.assertTrue(isinstance(env.observation_space, DiscreteSpace))
        self.assertTrue(env.observation_type == EnvObservationType.DISCRETE)
        self.assertTrue(isinstance(env.action_space, DiscreteSpace))

    def test_play_CartPole(self):
        # observation_space: Box
        # action_space     : Discrete
        env = self.tester.play_test("CartPole-v1")
        self.assertTrue(isinstance(env.observation_space, BoxSpace))
        self.assertTrue(env.observation_type == EnvObservationType.CONTINUOUS)
        self.assertTrue(isinstance(env.action_space, DiscreteSpace))

    def test_play_Blackjack(self):
        # observation_space: Tuple(Discrete(32), Discrete(11), Discrete(2))
        # action_space     : Discrete
        env = self.tester.play_test("Blackjack-v1")
        self.assertTrue(isinstance(env.observation_space, ArrayDiscreteSpace))
        self.assertTrue(env.observation_type == EnvObservationType.DISCRETE)
        self.assertTrue(isinstance(env.action_space, DiscreteSpace))

    def test_play_Pendulum(self):
        # observation_space: Box
        # action_space     : Box
        env = self.tester.play_test("Pendulum-v1")
        self.assertTrue(isinstance(env.observation_space, BoxSpace))
        self.assertTrue(env.observation_type == EnvObservationType.CONTINUOUS)
        self.assertTrue(isinstance(env.action_space, BoxSpace))

    def test_play_Pong(self):
        # atari
        env = self.tester.play_test("ALE/Pong-v5", check_render=False, max_step=100)
        self.assertTrue(isinstance(env.observation_space, BoxSpace))
        self.assertTrue(env.observation_type == EnvObservationType.UNKNOWN)
        self.assertTrue(isinstance(env.action_space, DiscreteSpace))

    def test_play_Tetris(self):
        # atari
        env = self.tester.play_test("ALE/Tetris-ram-v5", check_render=False, max_step=100)
        self.assertTrue(isinstance(env.observation_space, BoxSpace))
        self.assertTrue(env.observation_type == EnvObservationType.UNKNOWN)
        self.assertTrue(isinstance(env.action_space, DiscreteSpace))

    def test_play_all(self):
        import gym
        import gym.error
        from gym import envs
        from tqdm import tqdm

        specs = envs.registry.all()

        for spec in tqdm(list(reversed(list(specs)))):
            try:
                gym.make(spec.id)
                self.tester.play_test(spec.id, check_render=False, max_step=5)
            except AttributeError:
                pass
            except gym.error.DependencyNotInstalled:
                pass  # No module named 'mujoco_py'
            except ModuleNotFoundError:
                pass  # unsupported env
            except Exception:
                print(spec.id)
                raise


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_play_all", verbosity=2)

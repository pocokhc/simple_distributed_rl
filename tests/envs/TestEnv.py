import unittest

import numpy as np
import srl
from srl.base.define import EnvActionType, RenderType
from srl.base.env.base import EnvBase
from srl.runner import sequence
from srl.runner.callbacks import PrintProgress


class TestEnv:
    def __init__(self):
        pass

    def play_test(self, tester: unittest.TestCase, env_name: str):
        env = srl.envs.make(env_name)
        tester.assertTrue(issubclass(env.__class__, EnvBase))

        player_num = env.player_num
        tester.assertTrue(player_num > 0)

        # --- reset
        states, player_indexies = env.reset()
        tester.assertTrue(len(states) == player_num)
        for i in player_indexies:
            tester.assertTrue(0 <= i < player_num)
        for state in states:
            tester.assertTrue(isinstance(state, np.ndarray))

        # --- fetch_invalid_actions
        invalid_actions_list = env.fetch_invalid_actions()
        tester.assertTrue(len(invalid_actions_list) == player_num)
        for invalid_actions in invalid_actions_list:
            tester.assertTrue(isinstance(invalid_actions, list))

        # --- episode
        done = False
        total_reward = np.zeros(len(states))
        step = 0

        # render
        for mode in RenderType:
            try:
                env.render(mode)
            except NotImplementedError:
                pass

        while not done:
            # --- sample
            actions = env.sample()
            tester.assertTrue(len(actions) == player_num)
            if env.action_type == EnvActionType.DISCRETE:
                for i, action in enumerate(actions):
                    tester.assertTrue(0 <= action < env.action_space.n)
                    tester.assertTrue(action not in invalid_actions_list[i])

            # --- step
            states, rewards, player_indexies, done, info = env.step(actions, player_indexies)
            tester.assertTrue(len(states) == player_num)
            tester.assertTrue(len(rewards) == player_num)
            tester.assertTrue(isinstance(done, bool))
            tester.assertTrue(isinstance(info, dict))
            for i in player_indexies:
                tester.assertTrue(0 <= i < player_num)
            tester.assertTrue(len(player_indexies) == len(list(set(player_indexies))))
            for state in states:
                tester.assertTrue(isinstance(state, np.ndarray))
            for reward in rewards:
                tester.assertTrue(type(reward) in [int, float])

            total_reward += np.asarray(rewards)
            step += 1
            print(f"step {step}, action {actions}, reward {rewards}")

            # render
            for mode in RenderType:
                try:
                    env.render(mode)
                except NotImplementedError:
                    pass
        print(total_reward)

    def play_player(self, tester: unittest.TestCase, env_name: str, player: str):
        config = sequence.Config(
            env_name=env_name,
            rl_config=srl.rl.random_play.Config(),
        )
        env = config.make_env()
        config.players = [player] * env.player_num

        config.set_play_config(max_episodes=10, callbacks=[PrintProgress()])
        sequence.play(config)

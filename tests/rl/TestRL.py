import os
from typing import cast

import numpy as np
import srl.rl
from srl.base.define import EnvObservationType
from srl.base.env.singleplay_wrapper import SinglePlayerWrapper
from srl.envs.grid import Grid
from srl.rl.functions.common import to_str_observaten
from srl.rl.processor.image_processor import ImageProcessor  # noqa F401
from srl.runner import mp, sequence
from srl.runner.callbacks import PrintProgress

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class TestRL:
    def __init__(self):
        self.parameter = None
        self.config = None

    def play_test(self, tester, rl_config):

        env_list = [
            "FrozenLake-v1",
            "OneRoad",
            "OX",
        ]

        for env_name in env_list:
            config = sequence.Config(
                env_name=env_name,
                rl_config=rl_config,
            )
            with tester.subTest(("sequence", env_name, rl_config.getName())):
                self._sequence(config)

            with tester.subTest(("mp", env_name, rl_config.getName())):
                self._mp(config)

    def _sequence(self, config):
        # --- train
        config.set_train_config(max_steps=10)
        parameter, memory = sequence.train(config)

        # --- test
        config.set_play_config(max_episodes=1)
        episode_rewards, _, _ = sequence.play(config, parameter)

    def _mp(self, config):
        # --- train
        mp_config = mp.Config(worker_num=2)
        mp_config.set_train_config(max_train_count=10)
        parameter, memory = mp.train(config, mp_config)

        # --- test
        config.set_play_config(max_episodes=1)
        episode_rewards, _, _ = sequence.play(config, parameter)

    def play_verify_singleplay(
        self,
        tester,
        env_name,
        rl_config,
        train_count,
        test_episodes,
        is_atari=False,
    ):
        base_score = {
            "Grid": 0.65,  # 0.7318 ぐらい
            "2DGrid": 0.65,
            "Pendulum-v1": -500,  # -179.51776165585284ぐらい
            "IGrid": 1,  # 乱数要素なし
            "ALE/Pong-v5": 0.0,
        }
        assert env_name in base_score

        config = sequence.Config(
            env_name=env_name,
            rl_config=rl_config,
        )

        if is_atari:
            config.processors = [ImageProcessor(gray=True, resize=(84, 84), enable_norm=True)]
            config.max_episode_steps = 50
            config.override_env_observation_type = EnvObservationType.COLOR
            config.skip_frames = 4

        config.set_train_config(max_steps=train_count, callbacks=[PrintProgress(max_progress_time=10)])
        parameter, memory = sequence.train(config)

        config.set_play_config(max_episodes=test_episodes)
        episode_rewards, _, _ = sequence.play(config, parameter)
        s = f"{np.mean(episode_rewards)} >= {base_score[env_name]}"
        print(s)
        tester.assertTrue(np.mean(episode_rewards) >= base_score[env_name], s)

        self.parameter = parameter
        self.config = config

    def play_verify_2play(
        self,
        tester,
        env_name,
        rl_config,
        train_count,
        test_episodes,
        is_atari=False,
    ):
        base_score = {
            "OX": [0.8, 0.65],  # [0.987, 0.813] ぐらい
        }
        assert env_name in base_score

        config = sequence.Config(
            env_name=env_name,
            rl_config=rl_config,
            players=[None, None],
        )

        config.set_train_config(max_steps=train_count, callbacks=[PrintProgress(max_progress_time=10)])
        parameter, memory = sequence.train(config)

        # 2p random
        with tester.subTest("1p test"):
            config.players = [None, srl.rl.random_play.Config()]
            config.set_play_config(max_episodes=test_episodes)
            episode_rewards, _, _ = sequence.play(config, parameter)
            reward = np.mean([r[0] for r in episode_rewards])
            s = f"{reward} >= {base_score[env_name][0]}"
            print(s)
            tester.assertTrue(reward >= base_score[env_name][0], s)

        # 1p random
        with tester.subTest("2p test"):
            config.players = [srl.rl.random_play.Config(), None]
            config.set_play_config(max_episodes=test_episodes)
            episode_rewards, _, _ = sequence.play(config, parameter)
            reward = np.mean([r[1] for r in episode_rewards])
            s = f"{reward} >= {base_score[env_name][1]}"
            print(s)
            tester.assertTrue(reward >= base_score[env_name][1], s)

        self.parameter = parameter
        self.config = config

    def verify_grid_action_values(self, tester):
        assert self.config.env_name == "Grid"

        env = self.config.make_env()
        env_org = cast(Grid, env.get_original_env())
        V, Q = env_org.calc_action_values()
        for s, q in Q.items():
            q = list(q.values())
            rl_s = env.observation_encode(s)
            true_a = np.argmax(q)

            rl_q = self.parameter.get_action_values(rl_s, [])
            rl_a = np.argmax(rl_q)

            diff = abs(q[true_a] - rl_q[true_a])
            print(s, true_a, rl_a, diff)

            tester.assertTrue(true_a == rl_a)
            tester.assertTrue(diff < 0.2)

    def verify_grid_policy(self, tester):
        assert self.config.env_name == "Grid"

        env_for_rl = self.config.make_env()
        env = SinglePlayerWrapper(env_for_rl)
        env_org = cast(Grid, env.get_original_env())
        V, _Q = env_org.calc_action_values()
        Q = {}
        for k, v in _Q.items():
            new_k = env_for_rl.observation_encode(k)
            new_k = to_str_observaten(new_k)
            Q[new_k] = v

        worker = self.config.make_worker(self.parameter)
        worker.set_training(False)

        # 数ステップ回してactionを確認
        done = True
        for step in range(100):
            if done:
                state = env.reset()
                done = False
                total_reward = 0
                invalid_actions = env.fetch_invalid_actions()
                worker.on_reset(state, invalid_actions, env, [0])

            # action
            env_action, worker_action = worker.policy(state, invalid_actions, env, [0])
            assert env_action not in invalid_actions

            # -----------
            # policyのアクションと最適アクションが等しいか確認
            key = to_str_observaten(state)
            true_a = np.argmax(list(Q[key].values()))
            pred_a = env_for_rl.action_decode(env_action)
            print(f"{state}: {true_a} == {pred_a}")
            tester.assertTrue(true_a == pred_a)
            # -----------

            # env step
            next_state, reward, done, env_info = env.step(env_action)
            step += 1
            total_reward += reward
            next_invalid_actions = env.fetch_invalid_actions()

            # rl step
            work_info = worker.on_step(
                state, worker_action, next_state, reward, done, invalid_actions, next_invalid_actions, env
            )
            state = next_state

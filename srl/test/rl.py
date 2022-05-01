import os
from typing import cast

import numpy as np
import srl
from srl.base.define import EnvObservationType
from srl.base.env.processors import ImageProcessor
from srl.base.env.single_play_wrapper import SinglePlayerWrapper
from srl.envs.grid import Grid
from srl.rl.functions.common import to_str_observaten
from srl.runner import mp, sequence
from srl.runner.callbacks import PrintProgress, Rendering
from srl.runner.callbacks_mp import TrainFileLogger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class TestRL:
    def __init__(self):
        self.parameter = None
        self.config = None

        self.env_list = [
            srl.envs.Config("FrozenLake-v1"),
            srl.envs.Config("OneRoad"),
            srl.envs.Config("OX"),
        ]

        self.baseline = {
            # 1p
            "Grid": 0.65,  # 0.7318 ぐらい
            "2DGrid": 0.65,
            "Pendulum-v1": -500,  # -179.51776165585284ぐらい
            "IGrid": 1,  # 乱数要素なし
            "OneRoad": 1,  # 乱数要素なし
            "ALE/Pong-v5": 0.0,
            # 2p
            "OX": [0.8, 0.65],  # [0.987, 0.813] ぐらい
        }

    def play_sequence(self, rl_config):
        for env_config in self.env_list:
            config = sequence.Config(env_config, rl_config)

            # --- train
            config.set_train_config(max_steps=10, callbacks=[PrintProgress()])
            parameter, memory = sequence.train(config)

            # --- test
            config.set_play_config(max_episodes=1, callbacks=[Rendering()])
            episode_rewards, _, _ = sequence.play(config, parameter)

    def play_mp(self, rl_config_org):
        for env_config in self.env_list:
            rl_config = rl_config_org.copy()
            config = sequence.Config(env_config, rl_config)

            # --- train
            mp_config = mp.Config(worker_num=1)
            config.set_train_config()
            mp_config.set_train_config(
                max_train_count=10, callbacks=[TrainFileLogger(enable_checkpoint=False, enable_log=False)]
            )
            parameter, memory = mp.train(config, mp_config)

            # --- test
            config.set_play_config(max_episodes=1, callbacks=[Rendering()])
            episode_rewards, _, _ = sequence.play(config, parameter)

    def play_verify_singleplay(
        self,
        env_name,
        rl_config,
        train_count,
        test_episodes,
        is_atari=False,
    ):
        assert env_name in self.baseline

        env_config = srl.envs.Config(env_name)
        config = sequence.Config(env_config, rl_config)

        if is_atari:
            env_config.processors = [ImageProcessor(gray=True, resize=(84, 84), enable_norm=True)]
            env_config.override_env_observation_type = EnvObservationType.COLOR
            config.max_episode_steps = 50
            config.skip_frames = 4

        config.set_train_config(max_steps=train_count, callbacks=[PrintProgress(max_progress_time=10)])
        parameter, memory = sequence.train(config)

        config.set_play_config(max_episodes=test_episodes)
        episode_rewards, _, _ = sequence.play(config, parameter)
        s = f"{np.mean(episode_rewards)} >= {self.baseline[env_name]}"
        print(s)
        assert np.mean(episode_rewards) >= self.baseline[env_name], s

        self.parameter = parameter
        self.config = config

    def play_verify_2play(
        self,
        env_name,
        rl_config,
        train_count,
        test_episodes,
    ):
        assert env_name in self.baseline

        env_config = srl.envs.Config(env_name)
        config = sequence.Config(env_config, rl_config)
        config.players = [None, None]

        config.set_train_config(max_steps=train_count, callbacks=[PrintProgress(max_progress_time=10)])
        parameter, memory = sequence.train(config)

        # 2p random
        config.players = [None, srl.rl.random_play.Config()]
        config.set_play_config(max_episodes=test_episodes)
        episode_rewards, _, _ = sequence.play(config, parameter)
        reward = np.mean([r[0] for r in episode_rewards])
        s = f"{reward} >= {self.baseline[env_name][0]}"
        print(s)
        assert reward >= self.baseline[env_name][0], s

        # 1p random
        config.players = [srl.rl.random_play.Config(), None]
        config.set_play_config(max_episodes=test_episodes)
        episode_rewards, _, _ = sequence.play(config, parameter)
        reward = np.mean([r[1] for r in episode_rewards])
        s = f"{reward} >= {self.baseline[env_name][1]}"
        print(s)
        assert reward >= self.baseline[env_name][1], s

        self.parameter = parameter
        self.config = config

    def verify_grid_action_values(self):
        assert self.config.env_config.name == "Grid"

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

            assert true_a == rl_a
            assert diff < 0.2

    def verify_grid_policy(self):
        assert self.config.env_config.name == "Grid"

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
                state, invalid_actions = env.reset()
                done = False
                total_reward = 0
                worker.on_reset(state, invalid_actions, env)

            # action
            action = worker.policy(state, invalid_actions, env)
            assert action not in invalid_actions

            # -----------
            # policyのアクションと最適アクションが等しいか確認
            key = to_str_observaten(state)
            true_a = np.argmax(list(Q[key].values()))
            pred_a = env_for_rl.action_decode(action)
            print(f"{state}: {true_a} == {pred_a}")
            assert true_a == pred_a
            # -----------

            # env step
            state, reward, done, invalid_actions, env_info = env.step(action)
            step += 1

            # rl step
            worker.on_step(state, reward, done, invalid_actions, env)

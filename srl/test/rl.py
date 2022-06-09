import os
import warnings
from typing import cast

import numpy as np
import srl
from srl.base.define import EnvObservationType
from srl.base.env.singleplay_wrapper import SinglePlayEnvWrapper
from srl.base.rl.processors.image_processor import ImageProcessor
from srl.base.rl.registration import make_worker_rulebase
from srl.base.rl.singleplay_wrapper import SinglePlayWorkerWrapper
from srl.envs.grid import Grid
from srl.rl.functions.common import to_str_observation
from srl.runner import mp, sequence

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.simplefilter("ignore")


class TestRL:
    def __init__(self):
        self.parameter = None
        self.config: sequence.Config = None

        self.env_list = [
            srl.envs.Config("FrozenLake-v1"),
            srl.envs.Config("OX"),
        ]

        self.baseline = {
            # 1p
            "Grid": 0.65,  # 0.7318 ぐらい
            "2DGrid": 0.65,
            "Pendulum-v1": -500,  # -179.51776165585284ぐらい
            "PendulumImage-v0": -500,
            "IGrid": 1,  # 乱数要素なし
            "OneRoad": 1,  # 乱数要素なし
            "ALE/Pong-v5": 0.0,
            # 2p
            "StoneTaking": [1, 1],
            "OX": [0.8, 0.65],  # [0.987, 0.813] ぐらい
        }

    def play_sequence(self, rl_config):
        self._check_play_raw(rl_config)

        for env_config in self.env_list:
            config = sequence.Config(env_config, rl_config)

            # --- train
            parameter, memory, _ = sequence.train(config, max_steps=10, enable_validation=False, enable_history=False)

            # --- test
            sequence.evaluate(config, parameter, max_episodes=2, max_steps=10)

            # --- render
            sequence.render(config, parameter, max_steps=10)

    def _is_space_base_instance(self, val):
        if type(val) in [int, float, list, np.ndarray]:
            return True
        return False

    def _check_play_raw(self, rl_config):
        env_config = srl.envs.Config("OX")

        # --- init
        rl_config.assert_params()
        env = srl.envs.make(env_config)
        remote_memory, parameter, trainer, worker = srl.rl.make(rl_config, env)
        workers = [
            make_worker_rulebase("random"),
            worker,
        ]

        # --- episode
        worker.set_play_info(True, False)
        for _ in range(2):
            env.reset()
            [w.on_reset(env, i) for i, w in enumerate(workers)]
            assert worker.player_index == 1

            # --- step
            for step in range(10):
                # policy
                actions = [w.policy(env) for w in workers]
                if step % 2 == 0:
                    assert self._is_space_base_instance(actions[0])
                    assert actions[1] is None
                else:
                    assert actions[0] is None
                    assert self._is_space_base_instance(actions[1])

                if step == 0 or step == 1:
                    assert workers[0].info is None
                    assert workers[1].info is None
                elif step % 2 == 0:
                    assert isinstance(workers[0].info, dict)
                else:
                    assert isinstance(workers[1].info, dict)

                # render
                [w.render(env) for w in workers]

                # step
                env.step(actions)
                [w.on_step(env) for w in workers]

                if env.done:
                    assert isinstance(workers[0].info, dict)
                    assert isinstance(workers[1].info, dict)

                # train
                train_info = trainer.train()
                assert isinstance(train_info, dict)

                if env.done:
                    break

    def play_mp(self, rl_config_org):
        for env_config in self.env_list:
            rl_config = rl_config_org.copy()
            config = sequence.Config(env_config, rl_config)

            # --- train
            mp_config = mp.Config(worker_num=2)
            parameter, memory = mp.train(config, mp_config, max_train_count=5)

            # --- test
            sequence.evaluate(config, parameter, max_episodes=10, max_steps=10)

            # --- render
            sequence.render(config, parameter, max_steps=10)

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

        if env_name == "PendulumImage-v0":
            rl_config.override_env_observation_type = EnvObservationType.GRAY_2ch

        if is_atari:
            rl_config.override_env_observation_type = EnvObservationType.COLOR
            rl_config.processors = [ImageProcessor(gray=True, resize=(84, 84), enable_norm=True)]

        # create config
        config = sequence.Config(env_config, rl_config)
        if is_atari:
            config.max_episode_steps = 50
            config.skip_frames = 4

        parameter, memory, _ = sequence.train(
            config, max_steps=train_count, enable_validation=False, enable_history=False, max_progress_time=10
        )
        episode_rewards = sequence.evaluate(config, parameter, max_episodes=test_episodes)
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

        # self play training
        config.players = [None, None]
        parameter, memory, _ = sequence.train(
            config, max_steps=train_count, enable_validation=False, enable_history=False, max_progress_time=10
        )

        # 1p play
        config.players = [None, "random"]
        episode_rewards = sequence.evaluate(config, parameter, max_episodes=test_episodes)
        reward = np.mean([r[0] for r in episode_rewards])
        s = f"{reward} >= {self.baseline[env_name][0]}"
        print(s)
        assert reward >= self.baseline[env_name][0], s

        # 2p play
        config.players = ["random", None]
        episode_rewards = sequence.evaluate(config, parameter, max_episodes=test_episodes)
        reward = np.mean([r[1] for r in episode_rewards])
        s = f"{reward} >= {self.baseline[env_name][1]}"
        print(s)
        assert reward >= self.baseline[env_name][1], s

        self.parameter = parameter
        self.config = config

    def verify_grid_policy(self):
        assert self.config.env_config.name == "Grid"

        env_for_rl = self.config.make_env()
        env = SinglePlayEnvWrapper(env_for_rl)
        env_org = cast(Grid, env.get_original_env())

        worker = self.config.make_worker(self.parameter)
        worker.set_play_info(False, False)
        worker = SinglePlayWorkerWrapper(worker)

        V, _Q = env_org.calc_action_values()
        Q = {}
        for k, v in _Q.items():
            new_k = worker.worker.worker.observation_encode(k, env)
            new_k = to_str_observation(new_k)
            Q[new_k] = v

        # 数ステップ回してactionを確認
        done = True
        for _ in range(100):
            if done:
                state = env.reset()
                done = False
                worker.on_reset(env)

            # action
            action = worker.policy(env)

            # -----------
            # policyのアクションと最適アクションが等しいか確認
            key = to_str_observation(state)
            true_a = np.argmax(list(Q[key].values()))
            pred_a = worker.worker.worker.action_decode(action)
            print(f"{state}: {true_a} == {pred_a}")
            assert true_a == pred_a
            # -----------

            # env step
            state, reward, done, env_info = env.step(action)

            # rl step
            worker.on_step(env)

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
from srl.envs import grid, ox
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
            (srl.envs.Config("Grid"), grid.LayerProcessor()),
            (srl.envs.Config("FrozenLake-v1"), None),
            (srl.envs.Config("OX"), ox.LayerProcessor()),
        ]

        self.baseline = {
            # 1p
            "Grid": (0.65, 200),  # 0.7318 ぐらい
            "EasyGrid": (0.9, 100),  # 1
            "Pendulum-v1": (-500, 20),  # -179.51776165585284ぐらい
            "PendulumImage-v0": (-500, 10),
            "IGrid": (1, 200),  # 乱数要素なし
            "OneRoad": (1, 100),  # 乱数要素なし
            "ALE/Pong-v5": (0.0, 10),
            "Tiger": (0.5, 1000),
            # 2p
            "StoneTaking": ([0.9, 0.7], 200),  # 先行必勝(石10個)
            "OX": ([0.8, 0.65], 200),  # [0.987, 0.813] ぐらい
            "Othello4x4": ([0.1, 0.5], 200),  # [0.3, 0.9] ぐらい
        }

    def play_sequence(self, _rl_config, enable_image: bool = False):
        rl_config = _rl_config.copy()
        self._check_play_raw(rl_config, enable_image)

        for env_config, img_processor in self.env_list:
            config = sequence.Config(env_config, rl_config)

            if enable_image:
                if img_processor is None:
                    continue
                config.rl_config.processors = [img_processor]
            else:
                config.rl_config.processors = []

            # --- train
            parameter, memory, _ = sequence.train(
                config,
                max_steps=10,
                enable_validation=False,
                enable_file_logger=False,
            )

            # --- test
            sequence.evaluate(
                config,
                parameter,
                max_episodes=2,
                max_steps=10,
            )

            # --- render
            sequence.render(
                config,
                parameter,
                max_steps=10,
            )

    def _is_space_base_instance(self, val):
        if type(val) in [int, float, list, np.ndarray]:
            return True
        return False

    def _check_play_raw(self, rl_config, enable_image):
        env_config = srl.envs.Config("OX")
        if enable_image:
            rl_config.processors = [ox.LayerProcessor()]
        else:
            rl_config.processors = []
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

            # --- step
            for step in range(10):
                # policy
                action = workers[env.next_player_index].policy(env)
                assert self._is_space_base_instance(action)

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
                env.step(action)
                [w.on_step(env) for w in workers]

                if env.done:
                    assert isinstance(workers[0].info, dict)
                    assert isinstance(workers[1].info, dict)

                # train
                train_info = trainer.train()
                assert isinstance(train_info, dict)

                if env.done:
                    break

    def play_mp(self, _rl_config, enable_image: bool = False):
        rl_config = _rl_config.copy()

        for env_config, img_processor in self.env_list:
            config = sequence.Config(env_config, rl_config)

            if enable_image:
                if img_processor is None:
                    continue
                config.rl_config.processors = [img_processor]
            else:
                config.rl_config.processors = []

            # --- train
            mp_config = mp.Config(actor_num=2, allocate_trainer="/CPU:0")
            parameter, memory, _ = mp.train(
                config,
                mp_config,
                max_train_count=5,
                enable_file_logger=False,
            )

            # --- test
            sequence.evaluate(
                config,
                parameter,
                max_episodes=10,
                max_steps=10,
            )

            # --- render
            sequence.render(
                config,
                parameter,
                max_steps=10,
            )

    def play_verify_singleplay(
        self,
        env_name,
        _rl_config,
        train_count,
        test_num=0,
        is_atari=False,
        is_mp=False,
        is_valid: bool = False,
    ):
        assert env_name in self.baseline
        env_config = srl.envs.Config(env_name)
        rl_config = _rl_config.copy()

        if env_name == "PendulumImage-v0":
            rl_config.override_env_observation_type = EnvObservationType.GRAY_2ch

        # create config
        config = sequence.Config(env_config, rl_config)
        if is_atari:
            config.max_episode_steps = 50
            config.skip_frames = 4

        if is_mp:
            mp_config = mp.Config(1, allocate_trainer="/CPU:0")
            parameter, memory, _ = mp.train(
                config,
                mp_config,
                max_train_count=train_count,
                enable_validation=is_valid,
                enable_file_logger=False,
                max_progress_time=60,
            )
        else:
            parameter, memory, _ = sequence.train(
                config,
                max_steps=train_count,
                enable_validation=is_valid,
                enable_file_logger=False,
                max_progress_time=60,
            )

        true_env = self.baseline[env_name]
        if test_num == 0:
            max_episodes = true_env[1]
        else:
            max_episodes = test_num
        episode_rewards = sequence.evaluate(
            config,
            parameter,
            max_episodes=max_episodes,
            print_progress=True,
        )
        s = f"{np.mean(episode_rewards)} >= {true_env[0]}"
        print(s)
        assert np.mean(episode_rewards) >= true_env[0], s

        # parameter backup/restore
        param2 = config.make_parameter()
        param2.restore(parameter.backup())
        episode_rewards = sequence.evaluate(
            config,
            param2,
            max_episodes=max_episodes,
            print_progress=True,
        )
        s = f"backup/restore: {np.mean(episode_rewards)} >= {true_env[0]}"
        print(s)
        assert np.mean(episode_rewards) >= true_env[0], s

        self.parameter = parameter
        self.config = config

    def play_verify_2play(
        self,
        env_name,
        _rl_config,
        train_count,
        is_self_play: bool = True,
        is_valid: bool = False,
        is_mp=False,
    ):
        assert env_name in self.baseline
        rl_config = _rl_config.copy()

        env_config = srl.envs.Config(env_name)
        config = sequence.Config(env_config, rl_config)

        if is_self_play:
            config.players = [None, None]
        else:
            config.players = [None, "random"]

        if is_mp:
            mp_config = mp.Config(1, allocate_trainer="/CPU:0")
            parameter, memory, _ = mp.train(
                config,
                mp_config,
                max_train_count=train_count,
                enable_validation=is_valid,
                enable_file_logger=False,
                max_progress_time=10,
            )
        else:
            parameter, memory, _ = sequence.train(
                config,
                max_steps=train_count,
                enable_validation=is_valid,
                enable_file_logger=False,
                max_progress_time=10,
            )

        true_env = self.baseline[env_name]

        # 1p play
        config.players = [None, "random"]
        episode_rewards = sequence.evaluate(
            config,
            parameter,
            max_episodes=true_env[1],
            print_progress=True,
        )
        reward = np.mean([r[0] for r in episode_rewards])
        s = f"{reward} >= {true_env[0][0]}"
        print(s)
        assert reward >= true_env[0][0], s

        # 2p play
        config.players = ["random", None]
        episode_rewards = sequence.evaluate(
            config,
            parameter,
            max_episodes=true_env[1],
            print_progress=True,
        )
        reward = np.mean([r[1] for r in episode_rewards])
        s = f"{reward} >= {true_env[0][1]}"
        print(s)
        assert reward >= true_env[0][1], s

        # parameter backup/restore
        param2 = config.make_parameter()
        param2.restore(parameter.backup())
        episode_rewards = sequence.evaluate(
            config,
            param2,
            max_episodes=true_env[1],
            print_progress=True,
        )
        reward = np.mean([r[1] for r in episode_rewards])
        s = f"backup/restore: {reward} >= {true_env[0][1]}"
        print(s)
        assert reward >= true_env[0][1], s

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
            new_k = worker.worker.worker.state_encode(k, env)
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

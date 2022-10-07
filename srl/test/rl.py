from typing import Optional, cast

import numpy as np
import srl
from srl import runner
from srl.base.define import EnvObservationType
from srl.base.env.singleplay_wrapper import SinglePlayEnvWrapper
from srl.base.rl.singleplay_wrapper import SinglePlayWorkerWrapper
from srl.envs import grid, ox  # noqa F401
from srl.rl.functions.common import to_str_observation
from srl.utils.common import is_packages_installed


class TestRL:
    def __init__(self):
        self.parameter = None
        self.config: Optional[runner.Config] = None

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

    def simple_check(
        self,
        rl_config,
        env_name: str = "",
        enable_change_layer: bool = False,
        is_mp: bool = False,
        check_render: bool = True,
    ):
        if env_name == "":
            env_list = ["Grid", "OX"]
        else:
            env_list = [env_name]

        for env_name in env_list:

            env_config = srl.EnvConfig(env_name)
            rl_config = rl_config.copy(reset_env_config=True)

            if enable_change_layer:
                if env_name == "Grid":
                    rl_config.processors = [grid.LayerProcessor()]
                elif env_name == "OX":
                    rl_config.processors = [ox.LayerProcessor()]

            config = runner.Config(env_config, rl_config)

            if not is_mp:
                # --- check raw
                print(f"--- {env_name} raw check start ---")
                self._check_play_raw(env_config, rl_config, check_render)

                # --- check sequence
                print(f"--- {env_name} sequence check start ---")
                parameter, _, _ = runner.train(
                    config,
                    max_steps=10,
                    enable_evaluation=False,
                    enable_file_logger=False,
                )

                # --- check render
                if check_render:
                    runner.render(config, parameter, max_steps=10)
                    if is_packages_installed(["cv2", "matplotlib", "PIL", "pygame"]):
                        render = runner.animation(config, parameter, max_steps=10)
                        render.create_anime()

            else:
                print(f"--- {env_name} mp check start ---")
                mp_config = runner.MpConfig(actor_num=2)
                parameter, _, _ = runner.mp_train(
                    config,
                    mp_config,
                    max_train_count=10,
                    enable_evaluation=False,
                    enable_file_logger=False,
                )

            runner.evaluate(
                config,
                parameter,
                max_episodes=2,
                max_steps=10,
            )

    def simple_check_mp(self, rl_config, enable_change_layer: bool = False):
        self.simple_check(rl_config, enable_change_layer=enable_change_layer, is_mp=True)

    def _check_play_raw(self, env_config, rl_config, check_render):
        env = srl.make_env(env_config)
        rl_config.reset_config(env)
        rl_config.assert_params()

        parameter = srl.make_parameter(rl_config)
        remote_memory = srl.make_remote_memory(rl_config)
        trainer = srl.make_trainer(rl_config, parameter, remote_memory)
        workers = [srl.make_worker(rl_config, parameter, remote_memory, training=True) for _ in range(env.player_num)]

        # --- episode
        for _ in range(3):
            env.reset()
            [w.on_reset(env, i) for i, w in enumerate(workers)]

            # --- step
            for step in range(5):
                # policy
                action = workers[env.next_player_index].policy(env)
                assert env.action_space.check_val(action), f"Checking action_space failed. action={action}"

                for idx in range(env.player_num):
                    assert (workers[idx].info is None) or isinstance(
                        workers[idx].info, dict
                    ), f"unknown info type. worker{idx} info={workers[idx].info}"

                # render
                if check_render:
                    for w in workers:
                        w.render(env)

                # step
                env.step(action)
                [w.on_step(env) for w in workers]

                if env.done:
                    for idx in range(env.player_num):
                        assert isinstance(
                            workers[idx].info, dict
                        ), f"unknown info type. worker{idx} info={workers[idx].info}"

                # train
                train_info = trainer.train()
                assert isinstance(train_info, dict), f"unknown info type. train info={train_info}"

                if env.done:
                    break

    def verify_singleplay(
        self,
        env_name,
        _rl_config,
        train_count,
        test_num=0,
        is_atari=False,
        is_mp=False,
        is_eval: bool = False,
    ):
        assert env_name in self.baseline
        env_config = srl.EnvConfig(env_name)
        rl_config = _rl_config.copy()

        if env_name == "PendulumImage-v0":
            rl_config.override_env_observation_type = EnvObservationType.GRAY_2ch

        # create config
        config = runner.Config(env_config, rl_config)
        if is_atari:
            config.max_episode_steps = 50
            config.frameskip = 4

        if is_mp:
            mp_config = runner.MpConfig(1, allocate_trainer="/CPU:0")
            parameter, memory, _ = runner.mp_train(
                config,
                mp_config,
                max_train_count=train_count,
                enable_evaluation=is_eval,
                enable_file_logger=False,
                progress_max_time=60,
            )
        else:
            parameter, memory, _ = runner.train(
                config,
                max_steps=train_count,
                enable_evaluation=is_eval,
                enable_file_logger=False,
                progress_max_time=60,
            )

        true_env = self.baseline[env_name]
        if test_num == 0:
            max_episodes = true_env[1]
        else:
            max_episodes = test_num
        episode_rewards = runner.evaluate(
            config,
            parameter,
            max_episodes=max_episodes,
            print_progress=True,
        )
        s = f"{np.mean(episode_rewards)} >= {true_env[0]}"
        print(s)
        assert np.mean(episode_rewards) >= true_env[0], s

        # --- parameter backup/restore
        param2 = config.make_parameter()
        param2.restore(parameter.backup())
        episode_rewards = runner.evaluate(
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

    def verify_2play(
        self,
        env_name,
        _rl_config,
        train_count,
        is_self_play: bool = True,
        is_eval: bool = False,
        is_mp=False,
    ):
        assert env_name in self.baseline
        rl_config = _rl_config.copy()

        env_config = srl.EnvConfig(env_name)
        config = runner.Config(env_config, rl_config)

        if is_self_play:
            config.players = [None, None]
        else:
            config.players = [None, "random"]

        if is_mp:
            mp_config = runner.MpConfig(1, allocate_trainer="/CPU:0")
            parameter, memory, _ = runner.mp_train(
                config,
                mp_config,
                max_train_count=train_count,
                enable_evaluation=is_eval,
                enable_file_logger=False,
                progress_max_time=10,
            )
        else:
            parameter, memory, _ = runner.train(
                config,
                max_steps=train_count,
                enable_evaluation=is_eval,
                enable_file_logger=False,
                progress_max_time=10,
            )

        true_env = self.baseline[env_name]

        # 1p play
        config.players = [None, "random"]
        episode_rewards = runner.evaluate(
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
        episode_rewards = runner.evaluate(
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
        episode_rewards = runner.evaluate(
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
        from srl.envs.grid import Grid

        env_for_rl = self.config.make_env()
        env = SinglePlayEnvWrapper(env_for_rl)
        env_org = cast(Grid, env.get_original_env())

        worker = self.config.make_worker(self.parameter)
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
            key = to_str_observation(np.asarray(state))
            true_a = np.argmax(list(Q[key].values()))
            pred_a = worker.worker.worker.action_decode(action)
            print(f"{state}: {true_a} == {pred_a}")
            assert true_a == pred_a
            # -----------

            # env step
            state, reward, done, env_info = env.step(action)

            # rl step
            worker.on_step(env)

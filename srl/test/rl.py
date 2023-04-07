import unittest
from typing import List, Optional, Union, cast

import numpy as np

import srl
from srl import runner
from srl.base.env.config import EnvConfig
from srl.base.env.singleplay_wrapper import SinglePlayEnvWrapper
from srl.base.rl.config import RLConfig
from srl.base.rl.singleplay_wrapper import SinglePlayWorkerWrapper
from srl.envs import grid, ox  # noqa F401
from srl.rl.functions.common import to_str_observation
from srl.utils.common import is_packages_installed


class TestRL(unittest.TestCase):
    def __init__(self, *args):
        super().__init__(*args)

        self.rl_config = None
        self.simple_check_kwargs = {}

        self.baseline = {
            # 1p
            "Grid": 0.65,
            "EasyGrid": 0.9,
            "Pendulum-v1": -500,  # -179.51776165585284ぐらい
            "PendulumImage-v0": -500,
            "IGrid": 1,  # 乱数要素なし
            "OneRoad": 1,  # 乱数要素なし
            "ALE/Pong-v5": 0.0,
            "Tiger": 0.5,
            # 2p
            "StoneTaking": [0.9, 0.7],  # 先行必勝(石10個)
            "OX": [0.8, 0.65],  # [0.987, 0.813] ぐらい
            "Othello4x4": [0.1, 0.5],  # [0.3, 0.9] ぐらい
        }

    def test_simple_check(self):
        assert self.rl_config is not None, "Define `rl_config` with `setUp`."

        self.simple_check(self.rl_config, **self.simple_check_kwargs)

    def test_simple_check_mp(self):
        assert self.rl_config is not None, "Define `rl_config` with `setUp`."

        self.simple_check(self.rl_config, is_mp=True, **self.simple_check_kwargs)

    def test_summary(self):
        assert self.rl_config is not None, "Define `rl_config` with `setUp`."
        _rl_config = self.rl_config.copy(reset_env_config=True)

        env = srl.make_env("Grid")
        if self.simple_check_kwargs.get("use_layer_processor", False):
            _rl_config.processors.append(grid.LayerProcessor())
        parameter = srl.make_parameter(_rl_config, env)
        parameter.summary()

    # -----------------------------------

    def simple_check(
        self,
        rl_config: RLConfig,
        env_list: List[Union[str, EnvConfig]] = ["Grid", "OX"],
        is_mp: bool = False,
        train_kwargs: dict = dict(
            max_train_count=10,
            max_steps=-1,
            timeout=-1,
        ),
        use_layer_processor: bool = False,
        check_render: bool = True,
        enable_cpu: bool = True,
    ):
        env_list = env_list.copy()
        train_kwargs_ = dict(
            enable_evaluation=False,
            enable_file_logger=False,
        )
        train_kwargs_.update(train_kwargs)
        train_kwargs = train_kwargs_

        for env_config in env_list:
            env_config = srl.EnvConfig(env_config) if isinstance(env_config, str) else env_config.copy()
            _rl_config = rl_config.copy(reset_env_config=True)

            if use_layer_processor:
                if env_config.name == "Grid":
                    _rl_config.processors.append(grid.LayerProcessor())
                elif env_config.name == "OX":
                    _rl_config.processors.append(ox.LayerProcessor())

            config = runner.Config(env_config, _rl_config)
            if enable_cpu:
                config.allocate_main = "/CPU:0"
                config.allocate_trainer = "/CPU:0"
                config.allocate_actor = "/CPU:0"

            if not is_mp:
                # --- check raw
                print(f"--- {env_config.name} raw check start ---")
                self._check_play_raw(env_config, _rl_config, check_render)

                # --- check sequence
                print(f"--- {env_config.name} sequence check start ---")
                parameter, _, _ = runner.train(config, **train_kwargs)

                # --- check render
                if check_render:
                    runner.render(config, parameter, max_steps=10)
                    if is_packages_installed(["cv2", "matplotlib", "PIL", "pygame"]):
                        render = runner.animation(config, parameter, max_steps=10)
                        render.create_anime()

            else:
                print(f"--- {env_config.name} mp check start ---")
                config.actor_num = 2
                parameter, _, _ = runner.mp_train(config, **train_kwargs)

            runner.evaluate(
                config,
                parameter,
                max_episodes=2,
                max_steps=10,
            )

    def _check_play_raw(self, env_config, rl_config, check_render):
        env = srl.make_env(env_config)
        rl_config = rl_config.copy(reset_env_config=True)
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

    # ----------------------------------------------

    def verify(
        self,
        env_config: Union[str, EnvConfig],
        rl_config: RLConfig,
        train_count: int,
        train_kwargs: dict = {},
        train_players: List[Union[None, str, RLConfig]] = [None, "random"],
        evals=[
            dict(
                episode=100,
                players=[None, "random"],
                baseline=None,
                baseline_player=0,
            ),
        ],
        is_mp=False,
        train_steps: int = -1,
        check_restore: bool = True,
    ):
        # --- create config
        config = runner.Config(env_config, rl_config)

        # evals check
        env = config.make_env()
        for eval in evals:
            assert "episode" in eval
            assert "players" in eval
            baseline = eval.get("baseline", None)
            if baseline is None:
                baseline = env.reward_info.get("baseline", None)
            if baseline is None:
                baseline = self.baseline.get(env.config.name, None)
            assert baseline is not None, "Please specify a 'baseline'. (evals={})"
            eval["baseline"] = baseline

        # --- train
        config.players = train_players
        train_kwargs_ = dict(
            enable_evaluation=False,
            enable_file_logger=False,
            progress_max_time=60 * 5,
            max_train_count=train_count,
            max_steps=train_steps,
        )
        train_kwargs_.update(train_kwargs)
        train_kwargs = train_kwargs_
        if train_count <= 0:
            # 学習がいらないアルゴリズム
            train_kwargs["max_steps"] = train_steps
        if is_mp:
            parameter, memory, _ = runner.mp_train(config, **train_kwargs)
        else:
            parameter, memory, _ = runner.train(config, **train_kwargs)

        # --- eval
        for eval in evals:
            config.players = eval["players"]
            if check_restore:
                param2 = config.make_parameter()
                param2.restore(parameter.backup())
            else:
                param2 = parameter
            episode_rewards = runner.evaluate(
                config,
                param2,
                max_episodes=eval["episode"],
                print_progress=True,
            )
            baseline = eval["baseline"]
            if config.env_config.player_num == 1:
                reward = np.mean(episode_rewards)
            else:
                baseline_player = eval.get("baseline_player", 0)
                reward = np.mean([r[baseline_player] for r in episode_rewards])
                baseline = baseline[baseline_player]
            s = f"{reward} >= {baseline}"
            print(s)
            assert reward >= baseline, s

        return parameter

    def verify_1player(
        self,
        env_config: Union[str, EnvConfig],
        rl_config: RLConfig,
        train_count: int,
        train_kwargs: dict = {},
        eval_episode: int = 100,
        baseline: Optional[float] = None,
        is_mp=False,
        train_steps: int = -1,
        check_restore: bool = True,
    ):
        return self.verify(
            env_config,
            rl_config,
            train_count,
            train_kwargs,
            train_players=[None],
            evals=[
                dict(
                    episode=eval_episode,
                    players=[None],
                    baseline=baseline,
                    baseline_player=0,
                ),
            ],
            is_mp=is_mp,
            train_steps=train_steps,
            check_restore=check_restore,
        )

    def verify_2player(
        self,
        env_config: Union[str, EnvConfig],
        rl_config: RLConfig,
        train_count: int,
        train_kwargs: dict = {},
        train_players: List[Union[None, str, RLConfig]] = [None, "random"],
        eval_episode: int = 100,
        evals=[
            dict(
                players=[None, "random"],
                baseline=None,
                baseline_player=0,
            ),
            dict(
                players=["random", None],
                baseline=None,
                baseline_player=1,
            ),
        ],
        is_mp=False,
        train_steps: int = -1,
        check_restore: bool = True,
    ):
        for eval in evals:
            eval["episode"] = eval_episode
        return self.verify(
            env_config,
            rl_config,
            train_count,
            train_kwargs,
            train_players,
            evals,
            is_mp=is_mp,
            train_steps=train_steps,
            check_restore=check_restore,
        )

    # ----------------------------------------------------

    def verify_grid_policy(self, rl_config, parameter):
        from srl.envs.grid import Grid

        env_for_rl = srl.make_env("Grid")
        env = SinglePlayEnvWrapper(env_for_rl)
        env_org = cast(Grid, env.get_original_env())

        worker = srl.make_worker(rl_config, parameter)
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

from typing import List, Optional, Tuple, Union, cast

import numpy as np

import srl
from srl.base.env.config import EnvConfig
from srl.base.env.env_run import EnvRun
from srl.base.rl.config import DummyRLConfig, RLConfig
from srl.envs import grid, ox
from srl.rl.functions.common import to_str_observation
from srl.utils.common import is_available_pygame_video_device, is_packages_installed

BaseLineType = Union[float, List[Union[float, None]]]


class TestRL:
    def __init__(self):
        self.rl_config = None
        self.simple_check_kwargs = {}
        self.episode_baseline = {
            # (episode, baseline)
            # 1p
            "Grid": (200, 0.65),
            "EasyGrid": (100, 0.9),
            "Pendulum-v1": (10, -500),  # -179.51776165585284ぐらい
            "IGrid": (200, 1),  # 乱数要素なし
            "OneRoad": (100, 1),  # 乱数要素なし
            "ALE/Pong-v5": (5, 0.0),
            "Tiger": (1000, 0.5),
            # 2p(random)
            "StoneTaking": (200, [0.9, 0.7]),  # 先行必勝(石10個)
            "OX": (200, [0.8, 0.65]),  # [0.987, 0.813] ぐらい
            "Othello4x4": (50, [0.1, 0.5]),  # [0.3, 0.9] ぐらい
        }

    def simple_check(
        self,
        rl_config: RLConfig,
        env_list: List[Union[str, EnvConfig]] = ["Grid", "OX"],
        is_mp: bool = False,
        train_kwargs: dict = dict(
            max_train_count=2,
            max_steps=-1,
            timeout=-1,
        ),
        use_layer_processor: bool = False,
        check_render: bool = True,
    ):
        env_list = env_list.copy()
        train_kwargs_ = {}
        train_kwargs_.update(train_kwargs)

        rl_config.enable_assertion_value = True

        for env_config in env_list:
            rl_config2 = rl_config.copy(reset_env_config=True)
            env_config = srl.EnvConfig(env_config) if isinstance(env_config, str) else env_config.copy()

            if use_layer_processor:
                if env_config.name == "Grid":
                    rl_config2.processors.append(grid.LayerProcessor())
                elif env_config.name == "OX":
                    rl_config2.processors.append(ox.LayerProcessor())

            runner = srl.Runner(env_config, rl_config2)
            runner.set_device(device_trainer="CPU", device_actors="CPU")

            if not is_mp:
                # --- check sequence
                print(f"--- {env_config.name} sequence check start ---")
                runner.train(**train_kwargs_)
                assert runner.state.trainer is not None
                if train_kwargs_.get("max_train_count", 0) > 0:
                    assert runner.state.trainer.get_train_count() > 0

                # --- check render
                if check_render:
                    runner.render_terminal(max_steps=10)
                    if is_packages_installed(["cv2", "PIL", "pygame"]):
                        if is_available_pygame_video_device():
                            runner.render_window(max_steps=10, render_interval=1)
                        runner.animation_save_gif("_tmp.gif", max_steps=10)

                # --- check raw
                print(f"--- {env_config.name} raw check start ---")
                self.simple_check_raw(env_config, rl_config2, check_render)

            else:
                print(f"--- {env_config.name} mp check start ---")
                runner.train_mp(actor_num=2, **train_kwargs_)
            runner.evaluate(max_episodes=2, max_steps=10)

    def simple_check_raw(
        self,
        env_config: EnvConfig,
        rl_config: RLConfig,
        check_render: bool,
    ):
        env = srl.make_env(env_config)
        rl_config = rl_config.copy(reset_env_config=True)
        rl_config.enable_assertion_value = True
        rl_config.reset(env)
        rl_config.assert_params()

        parameter = srl.make_parameter(rl_config)
        remote_memory = srl.make_memory(rl_config)
        trainer = srl.make_trainer(rl_config, parameter, remote_memory)
        workers = [srl.make_worker(rl_config, env, parameter, remote_memory) for _ in range(env.player_num)]

        # --- episode
        for _ in range(3):
            env.reset()
            render_mode = "terminal" if check_render else ""
            [w.on_reset(i, training=True, render_mode=render_mode) for i, w in enumerate(workers)]

            # --- step
            for step in range(5):
                # policy
                action = workers[env.next_player_index].policy()
                assert env.action_space.check_val(action), f"Checking action_space failed. action={action}"

                for idx in range(env.player_num):
                    assert (workers[idx].info is None) or isinstance(
                        workers[idx].info, dict
                    ), f"unknown info type. worker{idx} info={workers[idx].info}"

                # render
                if check_render:
                    for w in workers:
                        w.render()

                # step
                env.step(action)
                [w.on_step() for w in workers]

                if env.done:
                    for idx in range(env.player_num):
                        assert isinstance(
                            workers[idx].info, dict
                        ), f"unknown info type. worker{idx} info={workers[idx].info}"

                    if check_render:
                        for w in workers:
                            w.render()

                # train
                trainer.train()
                assert isinstance(trainer.train_info, dict), f"unknown info type. train info={trainer.train_info}"

                if env.done:
                    break

    def simple_check_rulebase(
        self,
        name: str,
        env_list: List[Union[str, EnvConfig]] = ["Grid", "OX"],
        is_mp: bool = False,
        train_kwargs: dict = dict(
            max_train_count=-1,
            max_steps=10,
            timeout=-1,
        ),
        use_layer_processor: bool = False,
        check_render: bool = True,
        enable_gpu: bool = False,
    ):
        env_list = env_list.copy()
        train_kwargs_ = {}
        train_kwargs_.update(train_kwargs)

        for env_config in env_list:
            env_config = srl.EnvConfig(env_config) if isinstance(env_config, str) else env_config.copy()
            rl_config: RLConfig = DummyRLConfig(name=name)
            rl_config.enable_assertion_value = True

            if use_layer_processor:
                if env_config.name == "Grid":
                    rl_config.processors.append(grid.LayerProcessor())
                elif env_config.name == "OX":
                    rl_config.processors.append(ox.LayerProcessor())

            runner = srl.Runner(env_config, rl_config)
            runner.set_device(device_trainer="CPU", device_actors="CPU")
            if enable_gpu:
                runner.set_device(device_trainer="AUTO", device_actors="AUTO")

            if not is_mp:
                # --- check sequence
                print(f"--- {env_config.name} sequence check start ---")
                runner.train(**train_kwargs_)

                # --- check render
                if check_render:
                    runner.render_terminal(max_steps=10)
                    if is_packages_installed(["cv2", "PIL", "pygame"]):
                        if is_available_pygame_video_device():
                            runner.render_window(max_steps=10, render_interval=1)
                        runner.animation_save_gif("_tmp.gif", max_steps=10)

                # --- check raw
                print(f"--- {env_config.name} raw check start ---")
                self.simple_check_raw(env_config, rl_config, check_render)

            else:
                print(f"--- {env_config.name} mp check start ---")
                runner.train_mp(actor_num=2, **train_kwargs_)

            runner.evaluate(max_episodes=2, max_steps=10)

    # ----------------------------------------------
    def _check_baseline(
        self,
        env: EnvRun,
        env_name: str,
        episode: Optional[int],
        baseline: Optional[BaseLineType],
    ) -> Tuple[int, BaseLineType]:
        if baseline is None:
            baseline = env.reward_info.get("baseline", None)
        if baseline is None:
            if env.name in self.episode_baseline:
                baseline = self.episode_baseline[env_name][1]
        assert baseline is not None, "Please specify a 'baseline'."
        if isinstance(baseline, tuple):
            baseline = list(baseline)

        if episode is None:
            if env_name in self.episode_baseline:
                episode = self.episode_baseline[env_name][0]
        if episode is None:
            episode = 100

        return episode, baseline

    def eval(
        self,
        runner: srl.Runner,
        episode: Optional[int] = None,
        eval_kwargs: dict = {},
        baseline: Optional[BaseLineType] = None,
        check_restore: bool = True,
    ) -> Union[List[float], List[List[float]]]:  # single play , multi play
        env = runner.make_env()
        episode, baseline = self._check_baseline(env, env.name, episode, baseline)

        # --- check restore
        if check_restore:
            parameter = runner.make_parameter()
            parameter.restore(parameter.backup())

        # --- eval
        episode_rewards = runner.evaluate(max_episodes=episode, **eval_kwargs)

        # --- assert
        if not isinstance(baseline, list):
            reward = np.mean(episode_rewards)

            s = f"{reward} >= {baseline}"
            print(s)
            assert reward >= baseline, s
        else:
            rewards = np.mean(episode_rewards, axis=0)
            print(f"baseline {baseline}, rewards {rewards}")
            for i, reward in enumerate(rewards):
                if baseline[i] is not None:
                    assert reward >= baseline[i], f"{i} : {reward} >= {baseline[i]}"

        return episode_rewards

    def eval_2player(
        self,
        runner: srl.Runner,
        episode: Optional[int] = None,
        eval_1p_players=[None, "random"],
        eval_2p_players=["random", None],
        eval_kwargs: dict = {},
        baseline: Optional[BaseLineType] = None,
        check_restore: bool = True,
    ):
        # --- baseline check
        env = runner.make_env()
        episode, baseline = self._check_baseline(env, env.name, episode, baseline)
        assert isinstance(baseline, list)

        runner.set_players(eval_1p_players)
        self.eval(runner, episode, eval_kwargs, [baseline[0], None], check_restore)

        runner.set_players(eval_2p_players)
        self.eval(runner, episode, eval_kwargs, [None, baseline[1]], check_restore)

    # ----------------------------------------------------

    def verify_grid_policy(self, runner: srl.Runner):
        from srl.envs.grid import Grid

        env = srl.make_env("Grid")
        env_org = cast(Grid, env.get_original_env())
        worker = runner.make_worker()

        V, _Q = env_org.calc_action_values()
        Q = {}
        for k, v in _Q.items():
            new_k = worker.state_encode(k, env, append_recent_state=False)
            new_k = to_str_observation(new_k)
            Q[new_k] = v

        # 数ステップ回してactionを確認
        for _ in range(100):
            if env.done:
                env.reset()
                worker.on_reset(0, training=False)

            # action
            pred_a = worker.policy()

            # -----------
            # policyのアクションと最適アクションが等しいか確認
            key = to_str_observation(np.asarray(env.state))
            true_a = np.argmax(list(Q[key].values()))
            print(f"{env.state}: {true_a} == {pred_a}")
            assert true_a == pred_a
            # -----------

            # env step
            env.step(pred_a)

            # rl step
            worker.on_step()

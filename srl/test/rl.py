from typing import List, Optional, Tuple, Union, cast

import numpy as np

import srl
from srl import runner
from srl.base.env.config import EnvConfig
from srl.base.rl.base import RLParameter, RLRemoteMemory
from srl.base.rl.config import RLConfig
from srl.base.rl.worker import RLWorker
from srl.envs import grid, ox
from srl.rl.functions.common import to_str_observation
from srl.runner import Config
from srl.runner.callbacks.history_viewer import HistoryViewer
from srl.utils.common import is_packages_installed

BaseLineType = Union[float, List[Union[float, None]]]


class TestRL:
    #
    # ここは試行錯誤の末に、VSCodeで、pytestで継承先のみテストを表示し、
    # 継承元のテストを非表示にするテクニックです。
    # __init__ があるとpytestはテストとして認識しません。
    # __init__の引数は unittest 用です。
    # 継承先は class A(TestRL, unittest.TestCase): と継承します。
    # TestRL -> TestCase の順である必要があります。
    #
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rl_config = None
        self.simple_check_kwargs = {}
        self.baseline = {
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

    def init_simple_check(self) -> Tuple[RLConfig, dict]:
        raise NotImplementedError()

    def test_simple_check(self):
        self.init_simple_check()
        assert self.rl_config is not None, "Define `rl_config` with `init_simple_check`."
        self.simple_check(self.rl_config, **self.simple_check_kwargs)

    def test_simple_check_mp(self):
        self.init_simple_check()
        assert self.rl_config is not None, "Define `rl_config` with `init_simple_check`."

        self.simple_check(self.rl_config, is_mp=True, **self.simple_check_kwargs)

    def test_summary(self):
        self.init_simple_check()
        assert self.rl_config is not None, "Define `rl_config` with `init_simple_check`."
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
        enable_gpu: bool = False,
    ):
        env_list = env_list.copy()
        train_kwargs_ = dict(
            eval=None,
            history=None,
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
            config.device_main = "CPU"
            config.device_mp_trainer = "CPU"
            config.device_mp_actors = "CPU"
            if enable_gpu:
                config.device_main = "AUTO"
                config.device_mp_trainer = "GPU"

            if not is_mp:
                # --- check sequence
                print(f"--- {env_config.name} sequence check start ---")
                parameter, _, _ = runner.train(config, **train_kwargs)

                # --- check render
                if check_render:
                    runner.render(config, parameter, max_steps=10)
                    if is_packages_installed(["cv2", "matplotlib", "PIL", "pygame"]):
                        render = runner.animation(config, parameter, max_steps=10)
                        render.create_anime()

                # --- check raw
                print(f"--- {env_config.name} raw check start ---")
                self._check_play_raw(env_config, _rl_config, check_render)

            else:
                print(f"--- {env_config.name} mp check start ---")
                config.actor_num = 2
                parameter, _, _ = runner.train_mp(config, **train_kwargs)

            runner.evaluate(
                config,
                parameter,
                max_episodes=2,
                max_steps=10,
            )

    def _check_play_raw(self, env_config, rl_config, check_render):
        env = srl.make_env(env_config)
        rl_config = rl_config.copy(reset_env_config=True)
        rl_config.reset(env)
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
    def _check_baseline(self, config: Config, baseline: Optional[BaseLineType]) -> BaseLineType:
        env = config.make_env()
        if baseline is None:
            baseline = env.reward_info.get("baseline", None)
        if baseline is None:
            if config.env_config.name in self.baseline:
                baseline = self.baseline[config.env_config.name][1]
        assert baseline is not None, "Please specify a 'baseline'."
        return baseline

    def train_eval(
        self,
        config: runner.Config,
        train_count: int = -1,
        train_steps: int = -1,
        train_kwargs: dict = {},
        is_mp=False,
        eval_episode: Optional[int] = None,
        eval_kwargs: dict = {},
        baseline: Optional[BaseLineType] = None,
        check_restore: bool = True,
    ) -> Tuple[RLParameter, RLRemoteMemory, HistoryViewer]:
        baseline = self._check_baseline(config, baseline)
        assert config.env_config.player_num == 1, "For two or more players, use 'train' and 'eval'."

        parameter, memory, history = self.train(
            config,
            train_count,
            train_steps,
            train_kwargs,
            is_mp,
        )
        self.eval(
            config,
            parameter,
            eval_episode,
            eval_kwargs,
            baseline,
            check_restore,
        )
        return parameter, memory, history

    def train(
        self,
        config: runner.Config,
        train_count: int = -1,
        train_steps: int = -1,
        train_kwargs: dict = {},
        is_mp=False,
    ) -> Tuple[RLParameter, RLRemoteMemory, HistoryViewer]:
        assert train_count != -1 or train_steps != -1, "Please specify 'train_count' or 'train_steps'."
        config = config.copy(env_share=False)

        _train_kwargs = dict(
            eval=None,
            history=None,
            max_train_count=train_count,
            max_steps=train_steps,
        )
        _train_kwargs.update(train_kwargs)
        if is_mp:
            parameter, memory, history = runner.train_mp(config, **_train_kwargs)
        else:
            parameter, memory, history = runner.train(config, **_train_kwargs)

        return parameter, memory, history

    def eval(
        self,
        config: runner.Config,
        parameter: RLParameter,
        episode: Optional[int] = None,
        eval_kwargs: dict = {},
        baseline: Optional[BaseLineType] = None,
        check_restore: bool = True,
    ) -> Union[List[float], List[List[float]]]:  # single play , multi play
        config = config.copy(env_share=False)

        baseline = self._check_baseline(config, baseline)

        if episode is None:
            if config.env_config.name in self.baseline:
                episode = self.baseline[config.env_config.name][0]
        if episode is None:
            episode = 10

        # --- check restore
        if check_restore:
            param = config.make_parameter()
            param.restore(parameter.backup())
        else:
            param = parameter

        # --- eval
        _eval_kwargs = dict(
            max_episodes=episode,
        )
        _eval_kwargs.update(eval_kwargs)
        config.players
        episode_rewards = runner.evaluate(
            config,
            param,
            **_eval_kwargs,
        )

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
        config: runner.Config,
        parameter: RLParameter,
        episode: Optional[int] = None,
        eval_1p_players=[None, "random"],
        eval_2p_players=["random", None],
        eval_kwargs: dict = {},
        baseline: Optional[BaseLineType] = None,
        check_restore: bool = True,
    ):
        # --- baseline check
        baseline = self._check_baseline(config, baseline)
        assert isinstance(baseline, list)

        config.players = eval_1p_players
        self.eval(config, parameter, episode, eval_kwargs, [baseline[0], None], check_restore)

        config.players = eval_2p_players
        self.eval(config, parameter, episode, eval_kwargs, [None, baseline[1]], check_restore)

    # ----------------------------------------------------

    def verify_grid_policy(self, rl_config, parameter):
        from srl.envs.grid import Grid

        env = srl.make_env("Grid")
        env_org = cast(Grid, env.get_original_env())

        worker = srl.make_worker(rl_config, parameter, env=env)
        rl_worker = cast(RLWorker, worker.worker)

        V, _Q = env_org.calc_action_values()
        Q = {}
        for k, v in _Q.items():
            new_k = rl_worker.state_encode(k, env)
            new_k = to_str_observation(new_k)
            Q[new_k] = v

        # 数ステップ回してactionを確認
        for _ in range(100):
            if env.done:
                env.reset()
                worker.on_reset(env)

            # action
            pred_a = worker.policy(env)

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
            worker.on_step(env)

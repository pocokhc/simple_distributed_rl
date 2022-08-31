import time
import unittest
from typing import cast

import numpy as np
import srl
from algorithms import dqn
from envs import connectx
from srl import runner
from srl.base.define import EnvAction
from srl.base.env.base import EnvRun
from srl.base.rl.base import ExtendWorker, WorkerRun


class MyConnectXWorker(ExtendWorker):
    def __init__(self, *args):
        super().__init__(*args)

        # rlのconfig
        self.rl_config = cast(dqn.Config, self.rl_worker.worker.config)

        # MinMaxの探索数
        self.max_depth = 4

    def call_on_reset(self, env: EnvRun, worker_run: WorkerRun) -> None:
        self.is_rl = False
        self.scores = [0] * env.action_space.n
        self.minmax_time = 0
        self.minmax_count = 0

    def call_policy(self, env: EnvRun, worker_run: WorkerRun) -> EnvAction:
        if env.step_num == 0:
            # --- 先行1ターン目
            # DQNの探索率を0.5にして実行
            self.rl_config.epsilon = 0.5
            action = self.rl_worker.policy(env)
            self.is_rl = True
            return action

        # --- 2ターン目以降
        # DQNの探索率は0.1に戻す
        self.rl_config.epsilon = 0.1

        # 元の環境を取得
        env_org = cast(connectx.ConnectX, env.get_original_env())

        # MinMaxを実施、環境は壊さないようにcopyで渡す
        self.minmax_count = 0
        t0 = time.time()
        self.scores = self._minmax(env_org.copy())
        self.minmax_time = time.time() - t0

        # 最大スコア
        max_score = np.max(self.scores)
        max_count = np.count_nonzero(self.scores == max_score)

        # 最大数が1個ならそのアクションを実施
        if max_count == 1:
            self.is_rl = False
            action = int(np.argmax(self.scores))
            return action

        # 最大値以外のアクションを選択しないようにする(invalid_actionsに追加)
        new_invalid_actions = [a for a in range(env.action_space.n) if self.scores[a] != max_score]
        env.add_invalid_actions(new_invalid_actions, self.player_index)

        # rl実施
        action = self.rl_worker.policy(env)
        self.is_rl = True

        return action

    # MinMax
    def _minmax(self, env: "connectx.ConnectX", depth: int = 0):
        if depth == self.max_depth:
            return [0] * env.action_space.n

        self.minmax_count += 1

        # 有効なアクションを取得
        valid_actions = env.get_valid_actions(env.player_index)

        # env復元用に今の状態を保存
        env_dat = env.backup()

        if env.player_index == self.player_index:
            # 自分の番
            scores = [-9.0 for _ in range(env.action_space.n)]
            for a in valid_actions:
                # envを復元
                env.restore(env_dat)

                # env stepを実施
                _, r1, r2, done, _ = env.call_step(a)
                if done:
                    # 終了状態なら報酬をスコアにする
                    if self.player_index == 0:
                        scores[a] = r1
                    else:
                        scores[a] = r2
                else:
                    # 次のstepのスコアを取得
                    n_scores = self._minmax(env, depth + 1)
                    scores[a] = np.min(n_scores)  # 相手の番は最小を選択

        else:
            # 相手の番
            scores = [9.0 for _ in range(env.action_space.n)]
            for a in valid_actions:
                env.restore(env_dat)

                _, r1, r2, done, _ = env.call_step(a)
                if done:
                    if self.player_index == 0:
                        scores[a] = r1
                    else:
                        scores[a] = r2
                else:
                    n_scores = self._minmax(env, depth + 1)
                    scores[a] = np.max(n_scores)  # 自分の番は最大を選択

        return scores

    # 可視化用
    def call_render(self, env: EnvRun, worker_run: WorkerRun) -> None:
        print(f"- MinMax count: {self.minmax_count}, {self.minmax_time:.3f}s -")
        print("+---+---+---+---+---+---+---+")
        s = "|"
        for a in range(env.action_space.n):
            s += "{:2d} |".format(int(self.scores[a]))
        print(s)
        print("+---+---+---+---+---+---+---+")
        if self.is_rl:
            self.rl_worker.render(env)


class Test(unittest.TestCase):
    def setUp(self) -> None:
        env_config = srl.EnvConfig("ConnectX")
        rl_config = dqn.Config()
        rl_config.processors = [connectx.LayerProcessor()]
        rl_config.extend_worker = MyConnectXWorker
        self.config = runner.Config(env_config, rl_config)

    def test_run(self):
        # --- train
        self.config.players = [None, None]
        parameter, _, _ = runner.train(
            self.config,
            max_steps=100_000,
            enable_file_logger=False,
            enable_evaluation=False,
        )

        # --- eval
        env = self.config.make_env()
        org_env = cast(connectx.ConnectX, env.get_original_env())
        parameter = self.config.make_parameter()
        worker = self.config.make_worker(parameter)

        def my_agent(observation, configuration):
            step = observation.step

            # connectx は先行なら step==0、後攻なら step==1 がエピソードの最初
            if step == 0 or step == 1:
                env.direct_reset(observation, configuration)
                worker.on_reset(env, org_env.player_index)
            env.direct_step(observation, configuration)
            return worker.policy(env)

        import kaggle_environments  # pip install kaggle_environments

        kaggle_env = kaggle_environments.make("connectx", debug=True)
        players = []
        for players, p1_assert in [
            ([my_agent, "random"], 0.7),  # [ 1. -1.]
            (["random", my_agent], -0.7),  # [-1.  1.]
            ([my_agent, "negamax"], 0.3),  # [ 0.6 -0.6]
            (["negamax", my_agent], -0.3),  # [-0.6  0.6]
        ]:
            rewards = []
            for _ in range(100):
                steps = kaggle_env.run(players)
                rewards.append(steps[-1][0]["reward"])
            reward = np.mean(rewards)
            print(f"{players}: {abs(reward)} >= {abs(p1_assert)}")
            self.assertTrue(abs(reward) >= abs(p1_assert))


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_run", verbosity=2)

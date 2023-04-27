import time
from typing import Tuple, cast

import numpy as np

import srl
from srl import runner
from srl.algorithms import dqn
from srl.base.define import EnvAction
from srl.base.env.base import EnvRun
from srl.base.rl.worker import ExtendWorker, WorkerRun
from srl.envs import connectx


class MyConnectXWorker(ExtendWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.worker = cast(dqn.dqn.Worker, self.worker)

        # rlのconfig
        self.rl_config = cast(dqn.Config, self.worker.config)

        # MinMaxの探索数
        self.max_depth = 3

    def call_on_reset(self, env: EnvRun, worker_run: WorkerRun) -> dict:
        self._is_rl = False
        self.scores = [0] * env.action_space.n
        self.minmax_time = 0
        self.minmax_count = 0
        return {}

    def call_policy(self, env: EnvRun, worker_run: WorkerRun) -> Tuple[EnvAction, dict]:
        if env.step_num == 0:
            # --- 先行1ターン目
            # DQNの探索率を0.5にして実行
            self.rl_config.epsilon = 0.5
            action = self.rl_worker.policy(env)
            self._is_rl = True
            return action, {}

        # --- 2ターン目以降
        # DQNの探索率は0.1に戻す
        self.rl_config.epsilon = 0.1

        # MinMaxを実施、環境は壊さないようにcopyで渡す
        self.minmax_count = 0
        t0 = time.time()
        self.scores = self._minmax(env.copy())
        self.minmax_time = time.time() - t0

        # 最大スコア
        max_score = np.max(self.scores)
        max_count = np.count_nonzero(self.scores == max_score)

        # 最大数が1個ならそのアクションを実施
        if max_count == 1:
            self._is_rl = False
            action = int(np.argmax(self.scores))
            return action, {}

        # 最大値以外のアクションを選択しないようにする(invalid_actionsに追加)
        new_invalid_actions = [a for a in range(env.action_space.n) if self.scores[a] != max_score]
        env.add_invalid_actions(new_invalid_actions, self.player_index)

        # rl実施
        action = self.rl_worker.policy(env)
        self._is_rl = True

        return action, {}

    # --- MinMax
    # 探索にbackup/restoreを使っているので重い
    # EnvRunを使わずにboardをコピーして自作した方が早くなります
    def _minmax(self, env: EnvRun, depth: int = 0):
        if depth == self.max_depth:
            return [0] * env.action_space.n

        self.minmax_count += 1

        # 有効なアクションを取得
        valid_actions = env.get_valid_actions()

        # env復元用に今の状態を保存
        env_dat = env.backup()

        if env.next_player_index == self.player_index:
            # 自分の番
            scores = [-9.0 for _ in range(env.action_space.n)]
            for a in valid_actions:
                # envを復元
                env.restore(env_dat)

                # env stepを実施
                env.step(a)
                if env.done:
                    # 終了状態なら報酬をスコアにする
                    scores[a] = env.reward
                else:
                    # 次のstepのスコアを取得
                    n_scores = self._minmax(env, depth + 1)
                    scores[a] = np.min(n_scores)  # 相手の番は最小を選択

        else:
            # 相手の番
            scores = [9.0 for _ in range(env.action_space.n)]
            for a in valid_actions:
                env.restore(env_dat)

                env.step(a)
                if env.done:
                    scores[a] = env.reward
                else:
                    n_scores = self._minmax(env, depth + 1)
                    scores[a] = np.max(n_scores)  # 自分の番は最大を選択

        return scores

    # 可視化用
    def render_terminal(self, env: EnvRun, worker: WorkerRun, **kwargs) -> None:
        print(f"- MinMax count: {self.minmax_count}, {self.minmax_time:.3f}s -")
        print("+---+---+---+---+---+---+---+")
        s = "|"
        for a in range(env.action_space.n):
            s += "{:2d} |".format(int(self.scores[a]))
        print(s)
        print("+---+---+---+---+---+---+---+")
        if self._is_rl:
            self.rl_worker.render_terminal(env)


def create_config():
    env_config = srl.EnvConfig("connectx")

    rl_config = dqn.Config()
    rl_config.processors = [connectx.LayerProcessor()]

    # extend_workerに自作したクラスをいれます
    rl_config.extend_worker = MyConnectXWorker

    return runner.Config(env_config, rl_config)

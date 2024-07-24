import time
from typing import cast

import numpy as np

import srl
from srl.algorithms import dqn
from srl.base.env.env_run import EnvRun
from srl.base.rl.algorithms.extend_worker import ExtendWorker
from srl.envs import connectx


class MyConnectXWorker(ExtendWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.base_worker = cast(dqn.dqn.Worker, self.base_worker)

        # rlのconfig
        self.rl_config = cast(dqn.Config, self.base_worker.config)

        # MinMaxの探索数
        self.max_depth = 3

    def on_reset(self, worker):
        self.action_num = cast(connectx.ConnectX, worker.env.unwrapped).action_space.n
        self._is_rl = False
        self.scores = [0] * self.action_num
        self.minmax_time = 0
        self.minmax_count = 0

    def policy(self, worker) -> int:
        if worker.env.step_num == 0:
            # --- 先行1ターン目
            # DQNの探索率を0.5にして実行
            self.rl_config.epsilon = 0.5
            action = self.base_worker.policy(worker)
            self._is_rl = True
            return cast(int, action)

        # --- 2ターン目以降
        # DQNの探索率は0.1に戻す
        self.rl_config.epsilon = 0.1

        # MinMaxを実施、環境は壊さないようにcopyで渡す
        self.minmax_count = 0
        t0 = time.time()
        self.scores = self._minmax(self.worker.env.copy())
        self.minmax_time = time.time() - t0

        # 最大スコア
        max_score = np.max(self.scores)
        max_count = np.count_nonzero(self.scores == max_score)

        # 最大数が1個ならそのアクションを実施
        if max_count == 1:
            self._is_rl = False
            action = int(np.argmax(self.scores))
            return action

        # 最大値以外のアクションを選択しないようにする(invalid_actionsに追加)
        new_invalid_actions = [a for a in range(self.action_num) if self.scores[a] != max_score]
        worker.add_invalid_actions(new_invalid_actions)

        # rl実施
        action = self.base_worker.policy(worker)
        self._is_rl = True

        return cast(int, action)

    # --- MinMax
    # 探索にbackup/restoreを使っているので重い
    # EnvRunを使わずにboardをコピーして自作した方が早くなります
    def _minmax(self, env: EnvRun, depth: int = 0):
        if depth == self.max_depth:
            return [0] * self.action_num

        self.minmax_count += 1

        # 有効なアクションを取得
        valid_actions = env.get_valid_actions()

        # env復元用に今の状態を保存
        env_dat = env.backup()

        if env.next_player_index == self.player_index:
            # 自分の番
            scores = [-9.0 for _ in range(self.action_num)]
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
            scores = [9.0 for _ in range(self.action_num)]
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
    def render_terminal(self, worker, **kwargs) -> None:
        print(f"- MinMax count: {self.minmax_count}, {self.minmax_time:.3f}s -")
        print("+---+---+---+---+---+---+---+")
        s = "|"
        for a in range(self.action_num):
            s += "{:2d} |".format(int(self.scores[a]))
        print(s)
        print("+---+---+---+---+---+---+---+")
        if self._is_rl:
            self.base_worker.render_terminal(worker)


def create_runner():
    env_config = srl.EnvConfig("connectx", kwargs={"obs_type": "layer"})

    rl_config = dqn.Config()

    # extend_workerに自作したクラスをいれます
    rl_config.extend_worker = MyConnectXWorker

    return srl.Runner(env_config, rl_config)

import pickle
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
from sklearn.decomposition import PCA

from srl.base.define import DoneTypes
from srl.base.rl.worker_run import WorkerRun

if TYPE_CHECKING:
    from .config import Config
    from .torch_model import Model
    from .worker import Worker


@dataclass
class ArchiveCell:
    start_state: np.ndarray
    state: np.ndarray
    step: int
    reward: float
    backup: Optional[Any]
    #
    visit: int = 1
    select: int = 0

    def set(self, start_state, state, step, reward, backup):
        self.start_state = start_state
        self.state = state
        self.step = step
        self.reward = reward
        self.backup = backup


class Archive:
    def __init__(self, config: "Config", net: "Model"):
        self.config = config
        self.net = net

        self.archive_oz: List[np.ndarray] = []
        self.archive_cell: List[ArchiveCell] = []
        self.update_archive_count = 0
        self.update_archive_interval = 1
        self.novelty_threshold = config.archive_novelty_threshold
        self.updated = False

        self.l2_scale = np.sqrt(config.latent_size)
        self.restore_count = 0
        self.pca = None

    def backup(self):
        return pickle.dumps(
            [
                self.archive_cell,
                self.update_archive_count,
                self.update_archive_interval,
                self.novelty_threshold,
                self.updated,
            ]
        )

    def restore(self, data):
        d = pickle.loads(data)
        self.archive_cell = d[0]
        self.update_archive_count = d[1]
        self.update_archive_interval = d[2]
        self.novelty_threshold = d[3]
        self.updated = d[4]

        if len(self.archive_cell) > 0:
            states = np.stack([c.state for c in self.archive_cell])
            oe = self.net.encode_obs(states)
            oz = self.net.encode_latent(oe)
            self.archive_oz = []
            for i in range(len(self.archive_cell)):
                self.archive_oz.append(oz[i])

    def on_reset(self, worker: WorkerRun, rl_worker: "Worker"):
        self.is_backup = False
        self.episode_step = 0
        self.episode_reward = 0
        self.start_state = worker.state

        # --- 一定間隔で更新
        if self.update_archive_count > self.update_archive_interval:
            self.recalc_archive()
            # 閾値調整
            if len(self.archive_cell) > self.config.archive_max_size:
                for i in range(100):
                    self.novelty_threshold += self.config.archive_novelty_threshold
                    self.recalc_archive()
                    if len(self.archive_cell) < self.config.archive_max_size * 0.5:
                        break
                self.update_archive_interval = 0
                self.updated = True
            if self.updated:
                # 調整後に逆に少なくなったら閾値を下げる
                for i in range(10):
                    if len(self.archive_cell) > 10:
                        break
                    self.novelty_threshold /= 1.5
                    self.recalc_archive()
                    self.update_archive_interval = 0

            worker.info["archive_novel"] = self.novelty_threshold
            self.update_archive_count = 0
            self.update_archive_interval += 1
        self.update_archive_count += 1

        # --- 初期状態 check
        rl_worker.oe, rl_worker.q = self.net.pred_q(worker.state[np.newaxis, ...])
        rl_worker.q = rl_worker.q[0]
        rl_worker.oz = self.net.encode_latent(rl_worker.oe)
        start_oz = rl_worker.oz
        self.add_archive(rl_worker.oz, worker.state, worker.state, worker)

        # --- restore check
        if random.random() < self.config.archive_rate:
            cell = self._select_cell(start_oz)
            if cell is not None:
                cell.select += 1
                # step0はそのまま進める
                if cell.step > 0:
                    self.episode_step = cell.step
                    self.episode_reward = cell.reward
                    self.start_state = cell.start_state
                    worker.restore(cell.backup)
                    self.is_backup = True
                    self.restore_count += 1

        worker.info["restore"] = self.restore_count
        worker.info["archive"] = len(self.archive_cell)
        worker.info["archive_interval"] = self.update_archive_interval

    def _select_cell(self, start_oz: np.ndarray) -> Optional[ArchiveCell]:
        if len(self.archive_cell) == 0:
            return None

        # 開始位置からの距離を計算
        dists = np.linalg.norm(self.archive_oz - start_oz, axis=-1)
        sorted_indices = np.argsort(dists)

        # ランクベースでサンプリング
        self.rankbase_alpha = 1
        alpha = self.rankbase_alpha
        size = len(self.archive_oz)
        total = size * (2 + (size - 1) * alpha) / 2
        r = random.random() * total
        inverse_r = (alpha - 2 + np.sqrt((2 - alpha) ** 2 + 8 * alpha * r)) / (2 * alpha)
        idx = int(inverse_r)
        a_idx = sorted_indices[idx]

        return self.archive_cell[a_idx]

    def on_step(self, oz: np.ndarray, worker: WorkerRun) -> bool:
        self.episode_step += 1
        self.episode_reward += worker.reward

        # 終了状態は保存しない(abortは保存する)
        if worker.done and (worker.done_type != DoneTypes.ABORT):
            return False

        return self.add_archive(oz, self.start_state, worker.next_state, worker)

    def add_archive(self, oz: np.ndarray, start_state: np.ndarray, state: np.ndarray, worker: WorkerRun) -> bool:
        oz = oz.squeeze(0)
        _add = False
        if len(self.archive_cell) < self.config.archive_min_num:
            _add = True
        else:
            # 一番近いcell
            dists = np.linalg.norm(np.stack(self.archive_oz) - oz, axis=1)
            min_idx = np.argmin(dists)
            if dists[min_idx] / self.l2_scale > self.novelty_threshold:
                _add = True
        if _add:
            cell = ArchiveCell(
                start_state,
                state,
                self.episode_step,
                self.episode_reward,
                worker.backup(),
            )
            self.archive_oz.append(oz)
            self.archive_cell.append(cell)
            return True

        cell = self.archive_cell[min_idx]
        cell.visit += 1

        # --- update archive
        if self._is_update(cell, self.episode_reward, state):
            cell.set(
                start_state,
                state,
                self.episode_step,
                self.episode_reward,
                worker.backup(),
            )
            cell.select = 0
        return False

    def _is_update(self, cell: ArchiveCell, episode_reward: float, state):
        # restoreしたcellではupdateしない
        # 報酬が多ければ更新
        # 報酬が同じ場合は、内部距離が遠い方を採用
        if self.is_backup:
            return False
        if cell.reward < episode_reward:
            return True
        elif cell.reward == episode_reward:
            # 遠い方を優先
            ss = np.stack(
                [
                    cell.start_state,
                    cell.state,
                    state,
                ]
            )
            oes = self.net.encode_obs(ss)
            ozs = self.net.encode_latent(oes)

            start_oz = ozs[0]
            cell_oz = ozs[1]
            step_oz = ozs[2]

            d1 = np.linalg.norm(start_oz - cell_oz)
            d2 = np.linalg.norm(start_oz - step_oz)
            if d1 < d2:
                return True
        return False

    def recalc_archive(self):
        # 特徴を最新に更新
        states = np.stack([c.state for c in self.archive_cell])
        oe = self.net.encode_obs(states)
        oz = self.net.encode_latent(oe)
        for i in range(len(self.archive_cell)):
            self.archive_oz[i] = oz[i]

        # 近い状態同士はまとめる
        del_list = []
        for i in range(len(self.archive_oz) - 1):
            cell = self.archive_cell[i]
            # 自分と比較して一定以下のidxを追加
            dists = np.linalg.norm(np.stack(self.archive_oz[i + 1 :]) - self.archive_oz[i], axis=1) / self.l2_scale
            for j, d in enumerate(dists):
                dest_idx = i + j + 1
                if dest_idx in del_list:
                    continue
                if d < self.novelty_threshold:
                    del_list.append(dest_idx)
                    c2 = self.archive_cell[dest_idx]
                    if self._is_update(cell, c2.reward, c2.state):
                        cell.set(
                            c2.start_state,
                            c2.state,
                            c2.step,
                            c2.reward,
                            c2.backup,
                        )
                        cell.select = c2.select
                    cell.visit += c2.visit

        del_list = sorted(del_list, reverse=True)
        for i in del_list:
            del self.archive_cell[i]
            del self.archive_oz[i]

    def render_terminal(self, oz):
        print(f"archive size={len(self.archive_cell)}")
        if len(self.archive_cell) == 0:
            return

        dists = np.linalg.norm(np.stack(self.archive_oz) - oz, axis=1)
        min_idx = np.argmin(dists)
        cell = self.archive_cell[min_idx]
        print(f" dist   : {dists[min_idx]}, idx: {min_idx}")
        print(f" step   : {cell.step}")
        print(f" reward : {cell.reward}")
        print(f" select : {cell.select}")
        print(f" visit  : {cell.visit}")

    def render_rgb_array(self, screen, oz, base_x, base_y):
        if self.pca is None:
            if len(self.archive_oz) > 1:
                self.pca = PCA(n_components=2)
                self.archive_oz_2d = self.pca.fit_transform(np.asarray(self.archive_oz))
        if self.pca is None:
            return
        WIDTH = 200
        HEIGHT = 200
        PADDING = 20

        z_2d = self.pca.transform(oz)
        min_x = np.min(self.archive_oz_2d[:, 0])
        max_x = np.max(self.archive_oz_2d[:, 0])
        min_y = np.min(self.archive_oz_2d[:, 1])
        max_y = np.max(self.archive_oz_2d[:, 1])

        screen.draw_box(0, 0, WIDTH + PADDING * 2, HEIGHT + PADDING * 2, fill_color=(255, 255, 255))
        for i, (x, y) in enumerate(self.archive_oz_2d):
            screen_x = self._minmax_rescale(x, min_x, max_x, WIDTH)
            screen_y = self._minmax_rescale(y, min_y, max_y, HEIGHT)
            screen.draw_text(PADDING + base_x + screen_x, PADDING + base_y + screen_y, f"{i}", outline_width=1)
        screen_x = self._minmax_rescale(z_2d[0, 0], min_x, max_x, WIDTH)
        screen_y = self._minmax_rescale(z_2d[0, 1], min_y, max_y, HEIGHT)
        screen.draw_circle(PADDING + base_x + screen_x, PADDING + base_y + screen_y, 3, filled=True, fill_color=(255, 0, 0), line_color=(255, 0, 0))

    def _minmax_rescale(self, val: float, min_val: float, max_val: float, scale: float) -> float:
        """min_val〜max_val を 0〜scale にマッピング"""
        if max_val <= min_val:
            return val
        x = (val - min_val) / (max_val - min_val) * scale
        return min(max(x, 0), scale)

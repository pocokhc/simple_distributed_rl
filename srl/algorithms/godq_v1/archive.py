import pickle
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from srl.base.define import DoneTypes
from srl.base.rl.worker_run import WorkerRun
from srl.rl.functions import get_random_idx_by_rankbase

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
    dist: float = 0

    def set(self, start_state, state, step, reward, backup, dist):
        self.start_state = start_state
        self.state = state
        self.step = step
        self.reward = reward
        self.backup = backup
        self.dist = dist

    def __lt__(self, o: "ArchiveCell") -> bool:
        """ソート用の '<' の定義"""
        if self.reward < o.reward:
            return True
        if self.reward == o.reward and self.dist < o.dist:
            return True
        return False


class Archive:
    def __init__(self, config: "Config", net: "Model"):
        self.config = config
        self.net = net

        self.archive_cells: Dict[str, Dict[int, List[ArchiveCell]]] = {}
        self.restore_count = 0

    def backup(self):
        return pickle.dumps(self.archive_cells)

    def restore(self, data):
        d = pickle.loads(data)
        self.archive_cells = d

    def on_reset(self, worker: WorkerRun, rl_worker: "Worker"):
        if not worker.training:
            return

        self.episode_step = 0
        self.episode_reward = 0
        self.start_state = worker.state.astype(np.float32)
        self.start_state_str = self.config.observation_space.to_str(worker.state)

        # --- restore check
        if random.random() < self.config.archive_rate:
            cell = self._select_cell()
            if cell is not None:
                self.episode_step = cell.step
                self.episode_reward = cell.reward
                self.start_state = cell.start_state
                worker.restore(cell.backup)
                self.restore_count += 1
                worker.info["restore"] = self.restore_count

                # restore後のq
                rl_worker.oe = None
                rl_worker.q = None

    def _select_cell(self) -> Optional[ArchiveCell]:
        if len(self.archive_cells) == 0:
            return None
        if self.start_state_str not in self.archive_cells:
            return None
        steps = list(self.archive_cells[self.start_state_str].keys())
        if len(steps) == 0:
            return None

        # ランクベース
        steps.sort()
        idx = get_random_idx_by_rankbase(len(steps), self.config.archive_rankbase_alpha)
        r_step = steps[idx]

        # ランクベース
        cells = self.archive_cells[self.start_state_str][r_step]
        idx = get_random_idx_by_rankbase(len(cells), self.config.archive_rankbase_alpha)
        return cells[idx]

    def on_step(self, worker: WorkerRun, rl_worker: "Worker") -> bool:
        if not worker.training:
            return False

        self.episode_step += 1
        self.episode_reward += worker.reward

        # 一定ステップ毎が対象
        if (self.episode_step % self.config.archive_steps) != 0:
            return False

        # 終了状態は保存しない(abortは保存する)
        if worker.done and (worker.done_type != DoneTypes.ABORT):
            return False

        is_add = self._add_archive(worker.next_state.astype(np.float32), worker)
        worker.info["archive"] = self.get_archive_size()
        return is_add

    def get_archive_size(self) -> int:
        n = 0
        for v in self.archive_cells.values():
            for cells in v.values():
                n += len(cells)
        return n

    def _add_archive(self, state: np.ndarray, worker: WorkerRun) -> bool:
        # 1. 報酬が多い
        # 2. スタート地点から遠い
        if self.start_state_str not in self.archive_cells:
            self.archive_cells[self.start_state_str] = {}
        if self.episode_step not in self.archive_cells[self.start_state_str]:
            self.archive_cells[self.start_state_str][self.episode_step] = []
        dist = np.linalg.norm(self.start_state - state)
        new_c = ArchiveCell(
            self.start_state,
            state,
            self.episode_step,
            self.episode_reward,
            worker.backup(),
            float(dist),
        )
        if len(self.archive_cells[self.start_state_str][self.episode_step]) < self.config.archive_max_size:
            self.archive_cells[self.start_state_str][self.episode_step].append(new_c)
            self.archive_cells[self.start_state_str][self.episode_step].sort()
            return True
        else:
            c = self.archive_cells[self.start_state_str][self.episode_step][0]
            if c < new_c:
                del self.archive_cells[self.start_state_str][self.episode_step][0]
                self.archive_cells[self.start_state_str][self.episode_step].append(new_c)
                self.archive_cells[self.start_state_str][self.episode_step].sort()
                return True

        return False

    def find_nearest_archive(self, state: np.ndarray):
        arr = []
        for v in self.archive_cells.values():
            for cells in v.values():
                for c in cells:
                    dist = np.linalg.norm(c.state - state)
                    arr.append((float(dist), c))
        if len(arr) == 0:
            return None

        arr = sorted(arr, key=lambda x: x[0])
        return arr[0]

    def render_terminal(self, worker: WorkerRun):
        if self.start_state_str not in self.archive_cells:
            return
        print(f"archive size={self.get_archive_size()}")

        # 一番近いarchiveを取得
        dat = self.find_nearest_archive(worker.state)
        if dat is not None:
            dist = dat[0]
            cell: ArchiveCell = dat[1]
            print(f" dist  : {dist}")
            print(f" state : {str(cell.state)[:50]}")
            print(f" step  : {cell.step}")
            print(f" reward: {cell.reward}")

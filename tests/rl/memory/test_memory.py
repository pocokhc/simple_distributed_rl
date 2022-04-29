import collections
import math
import unittest

import numpy as np
from srl.rl.memory.proportional_memory import ProportionalMemory
from srl.rl.memory.rankbase_memory import RankBaseMemory
from srl.rl.memory.replay_memory import ReplayMemory


class TestMemory(unittest.TestCase):
    def test_memory(self):
        capacity = 10

        memories = [
            (ReplayMemory(capacity), False),
            (ProportionalMemory(capacity, 0.8, 1, 10), False),
            (RankBaseMemory(capacity, 0.8, 1, 10), False),
        ]
        for memory, use_priority in memories:
            with self.subTest(memory.__class__.__name__):
                self._test_memory(memory, use_priority)

    def _test_memory(self, memory, use_priority):
        capacity = memory.capacity
        assert capacity == 10

        # add
        for i in range(100):
            memory.add((i, i, i, i), 0)
        assert len(memory) == capacity

        # 中身を1～10にする
        for i in range(10):
            i += 1
            memory.add((i, i, i, i), i)
        assert len(memory) == capacity

        # --- 複数回やって比率をだす
        counter = []
        for i in range(10000):
            (indices, batchs, weights) = memory.sample(5, 1)
            assert len(indices) == 5
            assert len(batchs) == 5
            assert len(weights) == 5

            # 重複がないこと
            assert len(list(set(batchs))) == 5, list(set(batchs))

            # batchの中身をカウント
            for batch in batchs:
                counter.append(batch[0])

            # update priority
            memory.update(indices, batchs, [b[3] for b in batchs])
            assert len(memory) == capacity

            # save/load
            d = memory.backup()
            memory.restore(d)
            d2 = memory.backup()
            assert d == d2

        counter = collections.Counter(counter)

        # keyは1～10まであること
        keys = sorted(counter.keys())
        assert keys == [i + 1 for i in range(capacity)]

        if use_priority:
            # priorityが高いほど数が増えている
            vals = [counter[key] for key in keys]
            for i in range(capacity - 1):
                assert vals[i] < vals[i + 1]

    # -------------------------------
    # test IS
    # -------------------------------
    def test_IS_Proportional(self):
        capacity = 10
        for alpha in [0, 0.5, 0.8, 1.0]:
            with self.subTest(alpha=alpha):
                config = {
                    "capacity": capacity,
                    "alpha": alpha,
                    "beta_initial": 1,
                }
                memory = ProportionalMemory(**config)
                priorities = [
                    1**alpha,
                    2**alpha,
                    3**alpha,
                ]
                sum_priority = sum(priorities)
                probs = [
                    priorities[0] / sum_priority,
                    priorities[1] / sum_priority,
                    priorities[2] / sum_priority,
                ]
                self._test_IS(memory, priorities, probs)

    def test_IS_RankBase(self):
        capacity = 10
        for alpha in [0, 0.5, 0.8, 1.0]:
            with self.subTest(alpha=alpha):
                config = {
                    "capacity": capacity,
                    "alpha": alpha,
                    "beta_initial": 1,
                }
                memory = RankBaseMemory(**config)
                priorities = [
                    1 + 0 * alpha,  # 3位
                    1 + 1 * alpha,  # 2位
                    1 + 2 * alpha,  # 1位
                ]
                sum_priority = sum(priorities)
                probs = [
                    priorities[0] / sum_priority,
                    priorities[1] / sum_priority,
                    priorities[2] / sum_priority,
                ]
                self._test_IS(memory, priorities, probs)

    def _test_IS(self, memory, priorities, probs):
        for i in range(len(priorities)):
            memory.add((i, i, i, i + 1), i + 1)  # 最後がpriority

        N = len(priorities)
        true_weights = np.array(
            [
                (N * probs[0]) ** (-1),
                (N * probs[1]) ** (-1),
                (N * probs[2]) ** (-1),
            ]
        )
        maxw = np.max(true_weights)
        true_weights /= maxw

        (indices, batchs, weights) = memory.sample(3, 1)

        for i, b in enumerate(batchs):
            idx = b[0]
            self.assertTrue(
                math.isclose(weights[i], true_weights[idx], rel_tol=1e-7),
                f"{weights[i]} != {true_weights[idx]}",
            )


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="TestMemory.test_memory", verbosity=2)

import random
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import RLBaseTypes, SpaceTypes
from srl.base.rl.algorithms.base_dqn import RLConfig, RLWorker
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.rl import functions as funcs
from srl.rl.models.config.dueling_network import DuelingNetworkConfig
from srl.rl.models.config.input_config import RLConfigComponentInput
from srl.rl.tf.model import KerasModelAddedSummary

kl = keras.layers


@dataclass
class Config(
    RLConfig,
    RLConfigComponentInput,
):
    #: ε-greedy parameter for Test
    test_epsilon: float = 0
    epsilon: float = 0.01

    # --- archive parameters
    action_change_rate: float = 0.05
    explore_max_step: int = 100
    demo_batch_rate: float = 0.1
    w_visit: float = 0.3
    w_select: float = 0
    w_total_select: float = 0.1
    eps1: float = 0.001
    eps2: float = 0.00001

    # --- q parameters
    memory_warmup_size: int = 1_000
    memory_capacity: int = 10_000
    lr: float = 0.0005
    batch_size: int = 32
    #: Discount rate
    discount: float = 0.99
    #: Synchronization interval to Target network
    target_model_update_interval: int = 2000
    #: <:ref:`DuelingNetworkConfig`> hidden layer
    hidden_block: DuelingNetworkConfig = field(init=False, default_factory=lambda: DuelingNetworkConfig())

    def get_base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS | RLBaseTypes.IMAGE

    def get_framework(self) -> str:
        return "tensorflow"

    def get_name(self) -> str:
        return "Go-Explore"

    def assert_params(self) -> None:
        super().assert_params()
        self.assert_params_input()


register(
    Config(),
    __name__ + ":Memory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


class Memory(RLMemory[Config]):
    def __init__(self, *args):
        super().__init__(*args)

        self.archive = {}
        self.memory_q = []
        self.memory_demo = []

    def call_backup(self, **kwargs) -> Any:
        return [self.archive, self.memory_q, self.memory_demo]

    def call_restore(self, dat: Any, **kwargs) -> None:
        self.archive = dat[0]
        self.memory_q = dat[1]
        self.memory_demo = dat[2]

    def length(self):
        return len(self.memory_q)

    def add(self, mode: str, batch) -> None:
        if mode == "archive":
            self.archive_update(batch)
        elif mode == "q":
            self.memory_q.append(batch)
            if len(self.memory_q) > self.config.memory_capacity:
                self.memory_q.pop(0)
        elif mode == "demo":
            self.memory_demo.append(batch)
            if len(self.memory_demo) > self.config.memory_capacity:
                self.memory_demo.pop(0)

    # -----------------------------------------------

    def create_cell(self, state):
        space = self.config.observation_space
        if space.is_image():
            # (1) color -> gray
            if space.stype == SpaceTypes.COLOR:
                state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            elif space.stype == SpaceTypes.GRAY_3ch:
                state = np.squeeze(state, axis=-1)

            # (2) down sampling
            state = cv2.resize(state, (11, 8), interpolation=cv2.INTER_NEAREST)

            # (3) 255->8
            state = np.round(state * 8.0 / 255.0)

            return ",".join([str(int(n)) for n in state.flatten().tolist()])
        else:
            space.create_division_tbl(self.config.observation_division_num)
            return ",".join([str(n) for n in space.encode_to_list_int(state)])

    def archive_update(self, batch):
        state = batch[0]
        states = batch[1]
        actions = batch[2]
        rewards = batch[3]
        undone = batch[4]
        step = batch[5]
        total_reward = batch[6]
        backup = batch[7]

        cell_key = self.create_cell(state)
        if cell_key not in self.archive:
            self.archive[cell_key] = {
                "step": np.inf,
                "total_reward": -np.inf,
                "score": -np.inf,
                "visit": 0,
                "select": 0,
                "total_select": 0,
            }
        cell = self.archive[cell_key]
        cell["visit"] += 1

        # --- update archive
        _update = False
        if cell["total_reward"] < total_reward:
            _update = True
        elif (cell["total_reward"] == total_reward) and (cell["step"] > step):
            _update = True
        if _update:
            cell["score"] = self._calc_score(cell)
            cell["step"] = step
            cell["select"] = 0
            cell["total_reward"] = total_reward
            cell["states"] = states
            cell["actions"] = actions
            cell["rewards"] = rewards
            cell["undone"] = undone
            cell["backup"] = backup

    def _calc_score(self, cell):
        cnt_score1 = self.config.w_visit * (1 / (cell["visit"] + self.config.eps1)) + self.config.eps2
        cnt_score2 = self.config.w_select * (1 / (cell["select"] + self.config.eps1)) + self.config.eps2
        cnt_score3 = self.config.w_total_select * (1 / (cell["total_select"] + self.config.eps1)) + self.config.eps2
        neigh_score = 0
        level_weight = 1
        score = level_weight * (neigh_score + cnt_score1 + cnt_score2 + cnt_score3 + 1)
        return score

    def archive_select(self):
        if len(self.archive) == 0:
            return None

        # 累積和による乱数
        total = sum([c["score"] for c in self.archive.values()])
        if total == 0:
            return None
        r = random.random() * total
        n = 0
        for cell in self.archive.values():
            n += cell["score"]
            if r <= n:
                break

        cell["select"] += 1
        cell["total_select"] += 1
        cell["score"] = self._calc_score(cell)
        return cell

    # -----------------------------------------------

    def create_demo_memory(self):
        for cell in self.archive.values():
            for i in range(cell["step"]):
                batch = [
                    cell["states"][i],
                    cell["states"][i + 1],
                    cell["actions"][i],
                    cell["rewards"][i],
                    cell["undone"][i],
                ]
                self.memory_demo.append(batch)

    def sample_q(self, batch_size):
        return random.sample(self.memory_q, batch_size)

    def sample_demo(self, batch_size):
        return random.sample(self.memory_demo, batch_size)


class QNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        self.space = config.observation_space

        self.in_block = config.create_input_block_tf()
        self.hidden_block = config.hidden_block.create_block_tf(config.action_space.n)

        # build
        self(np.zeros((1,) + self.space.np_shape))
        self.loss_func = keras.losses.Huber()

    def to_state(self, state):
        state = self.space.encode_to_np(state, self.dtype)
        if self.space.is_image():
            state /= 255
        return state

    def call(self, x, training=False):
        x = self.in_block(x, training=training)
        x = self.hidden_block(x, training=training)
        return x

    @tf.function
    def compute_train_loss(self, state, onehot_action, target_q):
        q = self(state, training=True)
        q = tf.reduce_sum(q * onehot_action, axis=1)
        loss = self.loss_func(target_q, q)
        loss += tf.reduce_sum(self.losses)  # 正則化項
        return loss


class Parameter(RLParameter[Config]):
    def __init__(self, *args):
        super().__init__(*args)

        self.q_online = QNetwork(self.config, name="Q_online")
        self.q_target = QNetwork(self.config, name="Q_target")
        self.q_target.set_weights(self.q_online.get_weights())

    def call_restore(self, data, **kwargs):
        self.q_online.set_weights(data)
        self.q_target.set_weights(data)

    def call_backup(self, **kwargs):
        return self.q_online.get_weights()

    def summary(self, **kwargs):
        self.q_online.summary(**kwargs)


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def __init__(self, *args):
        super().__init__(*args)

        self.opt_q = keras.optimizers.Adam(self.config.lr)
        self.sync_count = 0

    def train(self) -> None:
        if len(self.memory.memory_q) < self.config.memory_warmup_size:
            return
        batchs = self.memory.sample_q(self.config.batch_size)
        state = []
        n_state = []
        action = []
        reward = []
        undone = []
        for b in batchs:
            state.append(b[0])
            n_state.append(b[1])
            action.append(b[2])
            reward.append(b[3])
            undone.append(b[4])
        demo_size = int(self.config.batch_size * self.config.demo_batch_rate)
        demo_size = demo_size if demo_size > 0 else 1
        if len(self.memory.memory_demo) > demo_size:
            batchs = self.memory.sample_demo(demo_size)
            for b in batchs:
                state.append(b[0])
                n_state.append(b[1])
                action.append(b[2])
                reward.append(b[3])
                undone.append(b[4])
        state = np.asarray(state, self.config.dtype)
        n_state = np.asarray(n_state, self.config.dtype)
        action = np.asarray(action, self.config.dtype)
        reward = np.array(reward, self.config.dtype)
        undone = np.array(undone, self.config.dtype)

        # --- calc next q
        batch_size = n_state.shape[0]
        n_q = self.parameter.q_online(n_state)
        n_q_target = self.parameter.q_target(n_state).numpy()
        n_act_idx = np.argmax(n_q, axis=-1)
        maxq = n_q_target[np.arange(batch_size), n_act_idx]
        target_q = reward + undone * self.config.discount * maxq
        target_q = target_q[..., np.newaxis]

        # --- train q
        with tf.GradientTape() as tape:
            loss = self.parameter.q_online.compute_train_loss(state, action, target_q)
        grad = tape.gradient(loss, self.parameter.q_online.trainable_variables)
        self.opt_q.apply_gradients(zip(grad, self.parameter.q_online.trainable_variables))
        self.info["loss"] = loss.numpy()

        # --- targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.set_weights(self.parameter.q_online.get_weights())
            self.sync_count += 1
            self.info["sync"] = self.sync_count

        self.train_count += 1


class Worker(RLWorker[Config, Parameter]):
    def on_start(self, worker, context):
        assert not self.distributed
        self.memory: Memory = self.memory

        if self.training and not self.rollout:
            self.memory.create_demo_memory()
            self.info["demo_size"] = len(self.memory.memory_demo)

    def on_reset(self, worker):
        if self.rollout:
            self.action = self.sample_action()
            batch = [
                worker.state,
                [self.parameter.q_online.to_state(worker.state)],
                [],
                [],
                [],
                0,
                0,
                worker.backup(),
            ]
            self.memory.add("archive", batch)

            self.cell_step = 0
            cell = self.memory.archive_select()
            if cell is not None:
                worker.restore(cell["backup"])
                self.episode_step = cell["step"]
                self.episode_reward = cell["total_reward"]
                self.recent_states = cell["states"][:]
                self.recent_actions = cell["actions"][:]
                self.recent_rewards = cell["rewards"][:]
                self.recent_undone = cell["undone"][:]
            else:
                self.episode_step = 0
                self.episode_reward = 0
                self.recent_states = [worker.state]
                self.recent_actions = []
                self.recent_rewards = []
                self.recent_undone = []

    def policy(self, worker) -> int:
        if self.rollout:
            if random.random() < self.config.action_change_rate:
                self.action = self.sample_action()
            return self.action
        elif self.training:
            epsilon = self.config.epsilon
        else:
            epsilon = self.config.test_epsilon

        if random.random() < epsilon:
            action = self.sample_action()
        else:
            state = worker.state[np.newaxis, ...]
            q = self.parameter.q_online(state)[0].numpy()
            action = int(np.argmax(q))
        return action

    def on_step(self, worker):
        if not self.training:
            return

        if self.rollout:
            self.episode_step += 1
            self.episode_reward += worker.reward
            self.recent_states.append(self.parameter.q_online.to_state(worker.state))
            self.recent_actions.append(funcs.one_hot(worker.action, self.config.action_space.n))
            self.recent_rewards.append(worker.reward)
            self.recent_undone.append(int(not worker.terminated))
            batch = [
                worker.state,
                self.recent_states[:],
                self.recent_actions[:],
                self.recent_rewards[:],
                self.recent_undone[:],
                self.episode_step,
                self.episode_reward,
                worker.backup(),
            ]
            self.memory.add("archive", batch)
            self.info["archive_size"] = len(self.memory.archive)

            self.cell_step += 1
            if self.cell_step > self.config.explore_max_step:
                worker.env.abort_episode()

        else:
            batch = [
                self.parameter.q_online.to_state(worker.prev_state),
                self.parameter.q_online.to_state(worker.state),
                funcs.one_hot(worker.action, self.config.action_space.n),
                worker.reward,
                int(not worker.terminated),
            ]
            self.memory.add("q", batch)

    def render_terminal(self, worker, **kwargs) -> None:
        # policy -> render -> env.step -> on_step

        # --- archive
        print(f"size: {len(self.memory.archive)}")
        key = self.memory.create_cell(worker.state)
        if key in self.memory.archive:
            cell = self.memory.archive[key]
            print(f"step        : {cell['step']}")
            print(f"total_reward: {cell['total_reward']}")
            print(f"score       : {cell['score']}")
            print(f"visit       : {cell['visit']}")
            print(f"select      : {cell['select']}")
            print(f"total_select: {cell['total_select']}")

        # --- q
        q = self.parameter.q_online(worker.state[np.newaxis, ...])[0]
        maxa = np.argmax(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        funcs.render_discrete_action(int(maxa), self.config.action_space, worker.env, _render_sub)

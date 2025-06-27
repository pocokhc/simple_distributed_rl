import copy
import pickle
import random
from typing import Any

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.rl.algorithms.base_dqn import RLWorker
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.rl.trainer import RLTrainer
from srl.rl import functions as funcs
from srl.rl.tf.model import KerasModelAddedSummary

from .config import Config

kl = keras.layers


class Memory(RLMemory[Config]):
    def setup(self) -> None:
        self.memory_q = []

        self.register_worker_func(self.add_q, pickle.dumps)
        self.register_trainer_recv_func(self.sample_q)

    def call_backup(self, **kwargs) -> Any:
        return self.memory_q[:]

    def call_restore(self, dat: Any, **kwargs) -> None:
        self.memory_q = dat

    def length(self):
        return len(self.memory_q)

    def add_q(self, batch, serialized: bool = False) -> None:
        if serialized:
            batch = pickle.loads(batch)
        self.memory_q.append(batch)
        if len(self.memory_q) > self.config.memory_capacity:
            self.memory_q.pop(0)

    def sample_q(self):
        if len(self.memory_q) < self.config.memory_warmup_size:
            return None
        return random.sample(self.memory_q, self.config.batch_size)


class QNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        self.space = config.observation_space

        if config.observation_space.is_value():
            self.in_block = config.input_value_block.create_tf_block(config.observation_space)
        elif config.observation_space.is_image():
            self.in_block = config.input_image_block.create_tf_block(config.observation_space)
        else:
            raise ValueError(config.observation_space)

        self.hidden_block = config.hidden_block.create_tf_block(config.action_space.n)

        # build
        self(np.zeros((1,) + self.space.np_shape))
        self.loss_func = keras.losses.Huber()

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
    def setup(self):
        self.q_online = QNetwork(self.config, name="Q_online")
        self.q_target = QNetwork(self.config, name="Q_target")
        self.q_target.set_weights(self.q_online.get_weights())
        self.archive = {}

    def call_restore(self, data, **kwargs):
        self.q_online.set_weights(data[0])
        self.q_target.set_weights(data[0])
        self.archive = copy.deepcopy(data[1])

    def call_backup(self, **kwargs):
        return [
            self.q_online.get_weights(),
            copy.deepcopy(self.archive),
        ]

    def summary(self, **kwargs):
        self.q_online.summary(**kwargs)

    def update_from_worker_parameter(self, worker_parameger: "Parameter"):
        self.archive = copy.deepcopy(worker_parameger.archive)

    # -------------------------------------

    def create_cell(self, state):
        return ",".join([str(int(n)) for n in state.flatten().tolist()])

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


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        self.opt_q = keras.optimizers.Adam(self.config.lr)
        self.sync_count = 0

        self.demo_buffer = []
        self.demo_size = int(self.config.batch_size * self.config.demo_batch_rate)
        self.demo_size = self.demo_size if self.demo_size > 0 else 1

        self.create_demo_memory()
        self.info["demo_size"] = len(self.demo_buffer)

    def create_demo_memory(self):
        for cell in self.parameter.archive.values():
            for i in range(cell["step"]):
                batch = [
                    cell["states"][i],
                    cell["states"][i + 1],
                    cell["actions"][i],
                    cell["rewards"][i],
                    cell["undone"][i],
                ]
                self.demo_buffer.append(batch)

    def train(self) -> None:
        batches = self.memory.sample_q()
        if batches is None:
            return

        state = []
        n_state = []
        action = []
        reward = []
        undone = []
        for b in batches:
            state.append(b[0])
            n_state.append(b[1])
            action.append(b[2])
            reward.append(b[3])
            undone.append(b[4])

        if len(self.demo_buffer) > self.demo_size:
            batches = random.sample(self.demo_buffer, self.demo_size)
            for b in batches:
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


class Worker(RLWorker[Config, Parameter, Memory]):
    def on_setup(self, worker, context):
        assert not self.distributed
        self.screen = None

    def on_reset(self, worker):
        if self.rollout:
            self.action = self.sample_action()
            """
            [
                render_image,
                states,
                actions,
                rewards,
                undone,
                step,
                total_reward,
                backup,
            ]
            """
            self.parameter.archive_update(
                [
                    worker.render_image_state,
                    [worker.state],
                    [],
                    [],
                    [],
                    0,
                    0,
                    worker.backup(),
                ]
            )

            self.cell_step = 0
            cell = self.parameter.archive_select()
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
            self.recent_states.append(worker.next_state)
            self.recent_actions.append(funcs.one_hot(worker.action, self.config.action_space.n))
            self.recent_rewards.append(worker.reward)
            self.recent_undone.append(int(not worker.terminated))
            """
            [
                render_image,
                states,
                actions,
                rewards,
                undone,
                step,
                total_reward,
                backup,
            ]
            """
            self.parameter.archive_update(
                [
                    worker.render_image_state,
                    self.recent_states[:],
                    self.recent_actions[:],
                    self.recent_rewards[:],
                    self.recent_undone[:],
                    self.episode_step,
                    self.episode_reward,
                    worker.backup(),
                ]
            )
            self.info["archive_size"] = len(self.parameter.archive)

            self.cell_step += 1
            if self.cell_step > self.config.explore_max_step:
                worker.env.abort_episode()

        else:
            self.memory.add_q(
                [
                    worker.state,
                    worker.next_state,
                    funcs.one_hot(worker.action, self.config.action_space.n),
                    worker.reward,
                    int(not worker.terminated),
                ]
            )

    def render_terminal(self, worker, **kwargs) -> None:
        # policy -> render -> env.step -> on_step

        # --- archive
        print(f"size: {len(self.parameter.archive)}")
        key = self.parameter.create_cell(worker.render_image_state)
        if key in self.parameter.archive:
            cell = self.parameter.archive[key]
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

        worker.print_discrete_action_info(int(maxa), _render_sub)

    def render_rgb_array(self, worker, **kwargs):
        # policy -> render -> env.step -> on_step
        from srl.utils import pygame_wrapper as pw

        IMG_W = 96
        IMG_H = 96
        PADDING = 2

        if self.screen is None:
            w = (IMG_W + PADDING) * 1 + 200
            h = (IMG_H + PADDING) * 4
            self.screen = pw.create_surface(w, h)
        pw.draw_fill(self.screen, color=(0, 0, 0))

        text_x = (IMG_W + PADDING) + 4
        img = worker.env.render_rgb_array()
        if img is None:
            img = worker.env.render_terminal_text_to_image()
        if img is None:
            return None
        pw.draw_image_rgb_array(self.screen, 0, 0, img, resize=(IMG_W, IMG_H))
        pw.draw_text(self.screen, text_x, 0, "original", color=(255, 255, 255))

        # (1) color -> gray
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        y = IMG_H + PADDING
        pw.draw_image_rgb_array(self.screen, 0, y, img, resize=(IMG_W, IMG_H), gray_to_color=True)
        pw.draw_text(self.screen, text_x, y, "gray", color=(255, 255, 255))

        # (2) down sampling
        img = cv2.resize(img, (self.config.downsampling_size[1], self.config.downsampling_size[0]), interpolation=cv2.INTER_NEAREST)
        y = (IMG_H + PADDING) * 2
        pw.draw_image_rgb_array(self.screen, 0, y, img, resize=(IMG_W, IMG_H), gray_to_color=True)
        pw.draw_text(self.screen, text_x, y, f"resize {self.config.downsampling_size}", color=(255, 255, 255))

        # (3) 255->8
        val = self.config.downsampling_val
        img = (np.round(img * val / 255.0) * 255 / val).astype(np.uint8)
        y = (IMG_H + PADDING) * 3
        pw.draw_image_rgb_array(self.screen, 0, y, img, resize=(IMG_W, IMG_H), gray_to_color=True)
        pw.draw_text(self.screen, text_x, y, f"255->{val}", color=(255, 255, 255))

        return pw.get_rgb_array(self.screen)

import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import SpaceTypes
from srl.base.rl.algorithms.base_dqn import RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.trainer import RLTrainer
from srl.base.rl.worker_run import WorkerRun
from srl.base.spaces.box import BoxSpace
from srl.rl import functions as funcs
from srl.rl.memories.replay_buffer import RLReplayBuffer
from srl.rl.tf.model import KerasModelAddedSummary

from .config import Config, DownSamplingProcessor

kl = keras.layers


class Memory(RLReplayBuffer):
    pass


class QNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        self.in_block = config.input_block.create_tf_block(config)
        self.hidden_block = config.hidden_block.create_tf_block(config.action_space.n)

        # build
        self(config.input_block.create_tf_dummy_data(config))
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
        self.q_ext_online = QNetwork(self.config)
        self.q_ext_target = QNetwork(self.config)
        self.q_ext_target.set_weights(self.q_ext_online.get_weights())

    def call_restore(self, data, **kwargs):
        self.q_ext_online.set_weights(data)
        self.q_ext_target.set_weights(data)

    def call_backup(self, **kwargs):
        return self.q_ext_online.get_weights()

    def summary(self, **kwargs):
        self.q_ext_online.summary(**kwargs)


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        self.opt_q_ext = keras.optimizers.Adam(self.config.lr_scheduler.apply_tf_scheduler(self.config.lr))
        self.sync_count = 0

    def train(self) -> None:
        batches = self.memory.sample()
        if batches is None:
            return
        state, n_state, action, reward, undone = zip(*batches)
        state = np.stack(state, axis=0)
        n_state = np.stack(n_state, axis=0)
        action = np.asarray(action, self.config.dtype)
        reward = np.array(reward, self.config.dtype)
        undone = np.array(undone, self.config.dtype)

        # --- calc next q
        n_q = self.parameter.q_ext_online(n_state)
        n_q_target = self.parameter.q_ext_target(n_state).numpy()
        n_act_idx = np.argmax(n_q, axis=-1)
        maxq = n_q_target[np.arange(self.config.batch_size), n_act_idx]
        target_q = reward + undone * self.config.discount * maxq
        target_q = target_q[..., np.newaxis]

        # --- train q
        with tf.GradientTape() as tape:
            loss = self.parameter.q_ext_online.compute_train_loss(state, action, target_q)
        grad = tape.gradient(loss, self.parameter.q_ext_online.trainable_variables)
        self.opt_q_ext.apply_gradients(zip(grad, self.parameter.q_ext_online.trainable_variables))
        self.info["loss_q"] = loss.numpy()

        # --- targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_ext_target.set_weights(self.parameter.q_ext_online.get_weights())
            self.sync_count += 1
            self.info["sync"] = self.sync_count

        self.train_count += 1


class Worker(RLWorker[Config, Parameter, Memory]):
    def __init__(self, *args):
        super().__init__(*args)
        self.screen = None

        self.archive = {}
        self.archive_total_visit = 0
        self.rmax = -np.inf
        self.rmin = np.inf

    def on_setup(self, worker, context):
        self.restore_count = 0

    def on_reset(self, worker):
        self.mode = ""
        self.q = None
        if not self.training:
            return

        self.episode_step = 0
        self.episode_reward = 0
        self.cell_step = 0

        if random.random() < self.config.go_rate:
            if random.random() < 0.5:
                self.mode = "go_random"
            else:
                self.mode = "go_action"
                self.go_action = self.sample_action()
            cell = self.archive_select()
            if cell is not None:
                self.episode_step = cell["step"]
                self.episode_reward = cell["total_reward"]
                backup = cell["backup"]
                if backup is not None:
                    worker.restore(backup)
                    self.restore_count += 1
        self.info["restore"] = self.restore_count
        self.info["archive"] = len(self.archive)

    def policy(self, worker) -> int:
        # invalid_actions = worker.invalid_actions

        if self.mode == "go_random":
            return self.sample_action()
        elif self.mode == "go_action":
            if random.random() < self.config.go_action_change_rate:
                self.go_action = self.sample_action()
            return self.go_action
        elif self.training:
            epsilon = self.config.epsilon
        else:
            epsilon = self.config.test_epsilon

        if random.random() < epsilon:
            action = self.sample_action()
        else:
            q = self.parameter.q_ext_online(worker.state[np.newaxis, ...])
            action = int(np.argmax(q[0]))

        return action

    def on_step(self, worker):
        if not self.training:
            return

        batch = [
            worker.state,
            worker.next_state,
            funcs.one_hot(worker.action, self.config.action_space.n),
            worker.reward,
            0 if self.worker.terminated else 1,
            # worker.invalid_actions,
            # worker.next_invalid_actions,
        ]
        self.memory.add(batch)

        if not worker.done:
            self.archive_update(worker)
            self.cell_step += 1
            if self.cell_step > self.config.search_max_step:
                worker.env.abort_episode()

    # ------------------------------

    def create_cell(self, state):
        return ",".join([str(int(n)) for n in state.flatten().tolist()])

    def archive_update(self, worker: WorkerRun):
        self.episode_step += 1
        self.episode_reward += worker.reward

        key = self.create_cell(worker.render_image_state)
        if key not in self.archive:
            self.archive[key] = {
                "step": np.inf,
                "total_reward": -np.inf,
                "visit": 0,
                "select": 0,
                "backup": None,
                "reward": 0,
            }
        cell = self.archive[key]
        cell["visit"] += 1
        self.archive_total_visit += 1

        self.rmin = min(self.rmin, worker.reward)
        self.rmax = max(self.rmax, worker.reward)

        # --- update archive
        _update = False
        if cell["total_reward"] < self.episode_reward:
            _update = True
        elif (cell["total_reward"] == self.episode_reward) and (cell["step"] > self.episode_step):
            _update = True
        if _update:
            cell["step"] = self.episode_step
            cell["total_reward"] = self.episode_reward
            cell["select"] = 0
            cell["backup"] = worker.backup()
            cell["reward"] = worker.reward

    def archive_select(self):
        if len(self.archive) == 0:
            return None
        if self.archive_total_visit == 0:
            return None

        max_ucb = -np.inf
        max_cells = []
        N = self.archive_total_visit
        for cell in self.archive.values():
            n = cell["visit"] + cell["select"]
            if n == 0:
                ucb = np.inf
            else:
                r = cell["reward"]
                if self.rmin < self.rmax:
                    r = (r - self.rmin) / (self.rmax - self.rmin)
                ucb = r * self.config.ucb_scale + np.sqrt(2 * np.log(N) / n)
            if max_ucb < ucb:
                max_ucb = ucb
                max_cells = [cell]
            elif max_ucb == ucb:
                max_cells.append(cell)
        max_cell = random.choice(max_cells)
        max_cell["select"] += 1

        return max_cell

    # -----------------------------------

    def render_terminal(self, worker, **kwargs) -> None:
        # policy -> render -> env.step -> on_step

        # --- archive
        print(f"size: {len(self.archive)}")
        key = self.create_cell(worker.render_image_state)
        if key in self.archive:
            cell = self.archive[key]
            print(f"step        : {cell['step']}")
            print(f"total_reward: {cell['total_reward']}")
            print(f"visit       : {cell['visit']}")
            print(f"select      : {cell['select']}")
            print(f"reward      : {cell['reward']}")

        # --- q
        q_ext = self.parameter.q_ext_online(worker.state[np.newaxis, ...]).numpy()[0]
        maxa = np.argmax(q_ext)

        def _render_sub(a: int) -> str:
            s = f"ext {q_ext[a]:7.5f}"
            return s

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

        img = worker.env.render_rgb_array()
        if img is None:
            img = worker.env.render_terminal_text_to_image()
        if img is None:
            return None

        processor = DownSamplingProcessor()
        img1, img2, img3 = processor.remap_observation(
            img,
            BoxSpace((1, 1, 3), stype=SpaceTypes.RGB),
            BoxSpace((1, 1, 3), stype=SpaceTypes.RGB),
            rl_config=self.config,
            _debug=True,
        )

        text_x = (IMG_W + PADDING) + 4
        y = (IMG_H + PADDING) * 0
        resize = (IMG_W, IMG_H)

        pw.draw_image_rgb_array(self.screen, 0, y, img1, resize=resize, gray_to_color=True)
        pw.draw_text(self.screen, text_x, y, "gray", color=(255, 255, 255))

        y = (IMG_H + PADDING) * 1
        pw.draw_image_rgb_array(self.screen, 0, y, img2, resize=(IMG_W, IMG_H), gray_to_color=True)
        pw.draw_text(self.screen, text_x, y, "255->2", color=(255, 255, 255))

        y = (IMG_H + PADDING) * 2
        pw.draw_image_rgb_array(self.screen, 0, y, img3 * 255, resize=(IMG_W, IMG_H), gray_to_color=True)
        pw.draw_text(self.screen, text_x, y, "resize", color=(255, 255, 255))

        return pw.get_rgb_array(self.screen)

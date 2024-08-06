import random
from dataclasses import dataclass, field
from typing import List, Union

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import SpaceTypes
from srl.base.env.env_run import EnvRun
from srl.base.rl.algorithms.base_dqn import RLConfig, RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.processor import RLProcessor
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.base.rl.worker_run import WorkerRun
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.space import SpaceBase
from srl.rl import functions as funcs
from srl.rl.memories.experience_replay_buffer import ExperienceReplayBuffer, RLConfigComponentExperienceReplayBuffer
from srl.rl.models.config.dueling_network import DuelingNetworkConfig
from srl.rl.models.config.input_config import RLConfigComponentInput
from srl.rl.schedulers.scheduler import SchedulerConfig
from srl.rl.tf.model import KerasModelAddedSummary

kl = keras.layers

# 初期値がランダムなDQN


class DownSamplingProcessor(RLProcessor):
    def remap_observation_space(self, env_observation_space: BoxSpace, env: EnvRun, rl_config: "Config") -> SpaceBase:
        self.space = env_observation_space
        return BoxSpace(rl_config.downsampling_size, 0, 8, np.uint8, SpaceTypes.GRAY_2ch)

    def remap_observation(self, img, worker: "Worker", env: EnvRun):
        if self.space.stype == SpaceTypes.COLOR:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif self.space.stype == SpaceTypes.GRAY_3ch:
            img = np.squeeze(img, axis=-1)

        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        img = cv2.resize(
            img,
            (worker.config.downsampling_size[1], worker.config.downsampling_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        return np.where(img < 127, 0, 1).astype(np.uint8)


@dataclass
class Config(
    RLConfig,
    RLConfigComponentInput,
    RLConfigComponentExperienceReplayBuffer,
):
    epsilon: float = 0.001
    test_epsilon: float = 0.00001

    restore_rate: float = 0.9

    #: <:ref:`scheduler`> Learning rate
    lr: Union[float, SchedulerConfig] = 0.0001
    batch_size: int = 64

    downsampling_size: tuple = (12, 12)

    go_rate: float = 0.9
    go_action_change_rate: float = 0.05
    ucb_scale: float = 0.1
    search_max_step: int = 100

    memory_warmup_size: int = 2_000
    memory_capacity: int = 50_000

    #: Discount rate
    discount: float = 0.995
    #: Synchronization interval to Target network
    target_model_update_interval: int = 2000

    #: <:ref:`MLPBlockConfig`> hidden layer
    hidden_block: DuelingNetworkConfig = field(init=False, default_factory=lambda: DuelingNetworkConfig())

    def get_processors(self) -> List[RLProcessor]:
        return RLConfigComponentInput.get_processors(self)

    def get_render_image_processors(self) -> List[RLProcessor]:
        return [DownSamplingProcessor()]

    def use_render_image_state(self) -> bool:
        return True

    def use_backup_restore(self) -> bool:
        return True

    def get_framework(self) -> str:
        return "tensorflow"

    def get_name(self) -> str:
        return "GoDQN"

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


class Memory(ExperienceReplayBuffer):
    pass


class QNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)

        self.space = config.observation_space

        self.in_block = config.create_input_block_tf()
        self.hidden_block = config.hidden_block.create_block_tf(config.action_space.n)

        # build
        self(np.zeros(self.in_block.create_batch_shape((1,))))
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
    def __init__(self, *args):
        super().__init__(*args)

        self.q_ext_online = QNetwork(self.config)
        self.q_ext_target = QNetwork(self.config)
        self.q_ext_target.set_weights(self.q_ext_online.get_weights())

    def call_restore(self, data, **kwargs):
        self.q_ext_online.set_weights(data[0])
        self.q_ext_target.set_weights(data[0])

    def call_backup(self, **kwargs):
        return [
            self.q_ext_online.get_weights(),
        ]

    def summary(self, **kwargs):
        self.q_ext_online.summary(**kwargs)


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def __init__(self, *args):
        super().__init__(*args)

        self.lr_sch = SchedulerConfig.create_scheduler(self.config.lr)
        self.opt_q_ext = keras.optimizers.Adam(self.lr_sch.get_rate())
        self.sync_count = 0

    def train(self) -> None:
        if self.memory.length() < self.config.memory_warmup_size:
            return
        batchs = self.memory.sample()
        state, n_state, action, reward, undone = zip(*batchs)
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


class Worker(RLWorker[Config, Parameter]):
    def __init__(self, *args):
        super().__init__(*args)
        self.screen = None

        self.archive = {}
        self.archive_total_visit = 0
        self.rmax = -np.inf
        self.rmin = np.inf

    def on_start(self, worker, context):
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
            worker.prev_state,
            worker.state,
            funcs.one_hot(worker.action, self.config.action_space.n),
            worker.reward,
            0 if self.worker.terminated else 1,
            # worker.prev_invalid_actions,
            # worker.invalid_actions,
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

        key = self.create_cell(worker.render_img_state)
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
        key = self.create_cell(worker.render_img_state)
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

        funcs.render_discrete_action(int(maxa), self.config.action_space, worker.env, _render_sub)

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

        # color -> gray
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        y = IMG_H + PADDING
        pw.draw_image_rgb_array(self.screen, 0, y, img, resize=(IMG_W, IMG_H), gray_to_color=True)
        pw.draw_text(self.screen, text_x, y, "gray", color=(255, 255, 255))

        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        y = (IMG_H + PADDING) * 2
        pw.draw_image_rgb_array(self.screen, 0, y, img, resize=(IMG_W, IMG_H), gray_to_color=True)
        pw.draw_text(self.screen, text_x, y, "255->2", color=(255, 255, 255))

        # down sampling
        img = cv2.resize(
            img, (self.config.downsampling_size[1], self.config.downsampling_size[0]), interpolation=cv2.INTER_NEAREST
        )
        y = (IMG_H + PADDING) * 3
        pw.draw_image_rgb_array(self.screen, 0, y, img, resize=(IMG_W, IMG_H), gray_to_color=True)
        pw.draw_text(self.screen, text_x, y, f"resize {self.config.downsampling_size}", color=(255, 255, 255))

        return pw.get_rgb_array(self.screen)

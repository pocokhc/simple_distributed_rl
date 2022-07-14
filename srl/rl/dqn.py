import random
from dataclasses import dataclass
from typing import Any, List, cast

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from srl.base.define import RLObservationType
from srl.base.env.base import EnvRun
from srl.base.rl.algorithms.discrete_action import DiscreteActionConfig, DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.registration import register
from srl.base.rl.remote_memory import ExperienceReplayBuffer
from srl.rl.functions.common import create_epsilon_list, inverse_rescaling, render_discrete_action, rescaling
from srl.rl.models.dqn_image_block import DQNImageBlock
from srl.rl.models.input_layer import create_input_layer
from srl.rl.models.mlp_block import MLPBlock
from tensorflow.keras import layers as kl

"""
Paper
・Playing Atari with Deep Reinforcement Learning
https://arxiv.org/pdf/1312.5602.pdf

・Human-level control through deep reinforcement learning
https://www.nature.com/articles/nature14236


window_length          : o (config selection)
Fixed Target Q-Network : o
Error clipping     : o
Experience Replay  : o
Frame skip         : -
Annealing e-greedy : o (config selection)
Reward clip        : o (config selection)
Image preprocessor : -

Other
    Double DQN               : o (config selection)
    Value function rescaling : o (config selection)
    invalid_actions          : o
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(DiscreteActionConfig):

    test_epsilon: float = 0

    epsilon: float = 0.1
    actor_epsilon: float = 0.4
    actor_alpha: float = 7.0

    # Annealing e-greedy
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.01
    exploration_steps: int = 0  # 0 : no Annealing

    # model
    window_length: int = 1
    cnn_block: kl.Layer = DQNImageBlock
    cnn_block_kwargs: dict = None
    hidden_block: kl.Layer = MLPBlock
    hidden_block_kwargs: dict = None

    gamma: float = 0.99  # 割引率
    lr: float = 0.001  # 学習率
    batch_size: int = 32
    capacity: int = 100_000
    memory_warmup_size: int = 1000
    target_model_update_interval: int = 1000
    enable_reward_clip: bool = False

    # other
    enable_double_dqn: bool = True
    enable_rescale: bool = False

    dummy_state_val: float = 0.0

    def set_config_by_actor(self, actor_num: int, actor_id: int) -> None:
        self.epsilon = create_epsilon_list(actor_num, epsilon=self.actor_epsilon, alpha=self.actor_alpha)[actor_id]

    # 論文のハイパーパラメーター
    def set_atari_config(self):
        self.batch_size = 32
        self.capacity = 1_000_000
        self.window_length = 4
        self.cnn_block = DQNImageBlock
        self.hidden_block = MLPBlock((512,))
        self.target_model_update_interval = 10000
        self.gamma = 0.99
        self.lr = 0.00025
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.1
        self.exploration_steps = 1_000_000
        self.memory_warmup_size = 50_000
        self.enable_reward_clip = True
        self.enable_double_dqn = False
        self.enable_rescale = False

    # -------------------------------

    def __post_init__(self):
        super().__init__()
        if self.cnn_block_kwargs is None:
            self.cnn_block_kwargs = {}
        if self.hidden_block_kwargs is None:
            self.hidden_block_kwargs = {}

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.CONTINUOUS

    @staticmethod
    def getName() -> str:
        return "DQN"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.window_length > 0
        assert self.memory_warmup_size < self.capacity
        assert self.batch_size < self.memory_warmup_size


register(
    Config,
    __name__ + ":RemoteMemory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


# ------------------------------------------------------
# RemoteMemory
# ------------------------------------------------------
class RemoteMemory(ExperienceReplayBuffer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.init(self.config.capacity)


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _QNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        in_state, c, use_image_head = create_input_layer(
            config.observation_shape,
            config.env_observation_type,
            config.window_length,
        )
        if use_image_head:
            c = config.cnn_block(**config.cnn_block_kwargs)(c)
            c = kl.Flatten()(c)

        # --- hidden block
        c = config.hidden_block(**config.hidden_block_kwargs)(c)

        # --- out layer
        c = kl.Dense(
            config.action_num,
            activation="linear",
            kernel_initializer="truncated_normal",
            bias_initializer="truncated_normal",
        )(c)
        self.model = keras.Model(in_state, c, name="QNetwork")

        # 重みを初期化
        dummy_state = np.zeros(shape=(1, config.window_length) + config.observation_shape, dtype=float)
        val = self(dummy_state)
        assert val.shape == (1, config.action_num)

    def call(self, state):
        return self.model(state)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.q_online = _QNetwork(self.config)
        self.q_target = _QNetwork(self.config)

    def restore(self, data: Any) -> None:
        self.q_online.set_weights(data)
        self.q_target.set_weights(data)

    def backup(self) -> Any:
        return self.q_online.get_weights()

    def summary(self, **kwargs):
        self.q_online.model.summary(**kwargs)


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

        self.optimizer = keras.optimizers.Adam(learning_rate=self.config.lr)
        self.loss = keras.losses.Huber()

        self.train_count = 0
        self.sync_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self):

        if self.remote_memory.length() < self.config.memory_warmup_size:
            return {}

        batchs = self.remote_memory.sample(self.config.batch_size)
        loss = self._train_on_batchs(batchs)

        # targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.set_weights(self.parameter.q_online.get_weights())
            self.sync_count += 1

        self.train_count += 1
        return {"loss": loss, "sync": self.sync_count}

    def _train_on_batchs(self, batchs):

        states = []
        actions = []
        n_states = []
        for b in batchs:
            states.append(b["states"][:-1])
            actions.append(b["action"])
            n_states.append(b["states"][1:])
        states = np.asarray(states)
        n_states = np.asarray(n_states)

        # next Q
        n_q = self.parameter.q_online(n_states).numpy()
        n_q_target = self.parameter.q_target(n_states).numpy()

        # 各バッチのQ値を計算
        target_q = np.zeros(len(batchs))
        for i, b in enumerate(batchs):
            reward = b["reward"]
            done = b["done"]
            next_invalid_actions = b["next_invalid_actions"]
            if done:
                gain = reward
            else:
                # DoubleDQN: indexはonlineQから選び、値はtargetQを選ぶ
                if self.config.enable_double_dqn:
                    n_q[i] = [(-np.inf if a in next_invalid_actions else v) for a, v in enumerate(n_q[i])]
                    n_act_idx = np.argmax(n_q[i])
                else:
                    n_q_target[i] = [
                        (-np.inf if a in next_invalid_actions else v) for a, v in enumerate(n_q_target[i])
                    ]
                    n_act_idx = np.argmax(n_q_target[i])
                maxq = n_q_target[i][n_act_idx]
                if self.config.enable_rescale:
                    maxq = inverse_rescaling(maxq)
                gain = reward + self.config.gamma * maxq
            if self.config.enable_rescale:
                gain = rescaling(gain)
            target_q[i] = gain

        with tf.GradientTape() as tape:
            q = self.parameter.q_online(states)

            # 現在選んだアクションのQ値
            actions_onehot = tf.one_hot(actions, self.config.action_num)
            q = tf.reduce_sum(q * actions_onehot, axis=1)

            loss = self.loss(target_q, q)

        grads = tape.gradient(loss, self.parameter.q_online.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.q_online.trainable_variables))

        return loss.numpy()


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(DiscreteActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

        self.dummy_state = np.full(self.config.observation_shape, self.config.dummy_state_val)
        self.step = 0

        if self.config.exploration_steps > 0:
            self.initial_epsilon = self.config.initial_epsilon
            self.epsilon_step = (
                self.config.initial_epsilon - self.config.final_epsilon
            ) / self.config.exploration_steps
            self.final_epsilon = self.config.final_epsilon

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> None:
        self.recent_states = [self.dummy_state for _ in range(self.config.window_length + 1)]

        self.recent_states.pop(0)
        self.recent_states.append(state)
        self.invalid_actions = invalid_actions

    def call_policy(self, _state: np.ndarray, invalid_actions: List[int]) -> int:

        if self.training:
            if self.config.exploration_steps > 0:
                # Annealing ε-greedy
                epsilon = self.initial_epsilon - self.step * self.epsilon_step
                if epsilon < self.final_epsilon:
                    epsilon = self.final_epsilon
            else:
                epsilon = self.config.epsilon
        else:
            epsilon = self.config.test_epsilon

        if random.random() < epsilon:
            # epsilonより低いならランダム
            action = random.choice([a for a in range(self.config.action_num) if a not in invalid_actions])
        else:
            state = self.recent_states[1:]
            q = self.parameter.q_online(np.asarray([state]))[0].numpy()

            # invalid actionsは -inf にする
            q = [(-np.inf if a in invalid_actions else v) for a, v in enumerate(q)]

            # 最大値を選ぶ（複数はほぼないので無視）
            action = int(np.argmax(q))

        self.action = action
        return action

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
    ):
        self.recent_states.pop(0)
        self.recent_states.append(next_state)
        self.invalid_actions = next_invalid_actions

        if not self.training:
            return {}
        self.step += 1

        # reward clip
        if self.config.enable_reward_clip:
            if reward < 0:
                reward = -1
            elif reward > 0:
                reward = 1
            else:
                reward = 0

        batch = {
            "states": self.recent_states[:],
            "action": self.action,
            "reward": reward,
            "done": done,
            "next_invalid_actions": next_invalid_actions,
        }
        self.remote_memory.add(batch)

        return {}

    def call_render(self, env: EnvRun) -> None:
        state = self.recent_states[1:]
        q = self.parameter.q_online(np.asarray([state]))[0].numpy()
        maxa = np.argmax(q)
        if self.config.enable_rescale:
            q = inverse_rescaling(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        render_discrete_action(self.invalid_actions, maxa, env, _render_sub)

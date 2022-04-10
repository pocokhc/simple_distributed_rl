import collections
import logging
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from srl.base.rl import RLParameter, RLRemoteMemory, RLTrainer, RLWorker
from srl.base.rl.config import DiscreteActionConfig
from srl.rl.functions.common import (
    calc_epsilon_greedy_probs,
    create_beta_list,
    create_epsilon_list,
    create_gamma_list,
    inverse_rescaling,
    rescaling,
)
from srl.rl.functions.dueling_network import create_dueling_network_layers
from srl.rl.functions.model import ImageLayerType, create_input_layers
from srl.rl.memory import factory
from srl.rl.registory import register
from tensorflow.keras import layers as kl

logger = logging.getLogger(__name__)

"""
DQN
    window_length       : o (config selection)
    Target Network      : o
    Huber loss function : o
    Delay update Target Network: o
    Experience Replay   : o
    Frame skip          : -
    Annealing e-greedy  : x
    Reward clip         : x
    Image preprocessor  : -
Rainbow
    Double DQN               : o (config selection)
    Priority Experience Reply: o (config selection)
    Dueling Network          : o (config selection)
    Multi-Step learning      : x
    Noisy Network            : x
    Categorical DQN          : x
Recurrent Replay Distributed DQN(R2D2)
    LSTM                     : x
    Value function rescaling : o
Never Give Up(NGU)
    Intrinsic Reward : o
    UVFA             : TODO
    Retrace          : x
Agent57
    Meta controller(sliding-window UCB) : o
    Intrinsic Reward split              : o
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(DiscreteActionConfig):

    # test
    test_epsilon: float = 0.0001
    test_beta: float = 0

    # model
    window_length: int = 1
    hidden_layer_sizes: Tuple[int, ...] = (512,)
    activation: str = "relu"
    image_layer_type: ImageLayerType = ImageLayerType.DQN
    batch_size: int = 32
    q_ext_lr: float = 0.001
    q_int_lr: float = 0.001
    target_model_update_interval: int = 100

    # double dqn
    enable_double_dqn: bool = True

    # DuelingNetwork
    enable_dueling_network: bool = True
    dueling_network_type: str = "average"
    dueling_dense_units: int = 512

    # Priority Experience Replay
    capacity: int = 100_000
    memory_name: str = "RankBaseMemory"
    memory_warmup_size: int = 1000
    memory_alpha: float = 0.6
    memory_beta_initial: float = 0.4
    memory_beta_steps: int = 1_000_000

    # ucb(160,0.5 or 3600,0.01)
    actor_num: int = 32
    ucb_window_size: int = 160  # UCB上限
    ucb_epsilon: float = 0.5  # UCBを使う確率
    ucb_beta: float = 1  # UCBのβ

    # episodic
    episodic_lr: float = 0.0005
    episodic_count_max: int = 10  # k
    episodic_epsilon: float = 0.001
    episodic_cluster_distance: float = 0.008
    episodic_memory_capacity: int = 30000
    episodic_pseudo_counts: float = 0.1  # 疑似カウント定数

    # lifelong
    lifelong_lr: float = 0.00001
    lifelong_max: float = 5.0  # L

    dummy_state_val: float = 0.0

    @staticmethod
    def getName() -> str:
        return "Agent57_light"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.window_length > 0


register(Config, __name__)


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _QNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        in_state, c = create_input_layers(
            config.window_length,
            config.env_observation_shape,
            config.env_observation_type,
            config.image_layer_type,
        )

        for h in config.hidden_layer_sizes:
            c = kl.Dense(h, activation=config.activation, kernel_initializer="he_normal")(c)

        if config.enable_dueling_network:
            c = create_dueling_network_layers(
                c, config.nb_actions, config.dueling_dense_units, config.dueling_network_type
            )
        else:
            c = kl.Dense(config.nb_actions, kernel_initializer="truncated_normal")(c)

        self.model = keras.Model(in_state, c)

        # 重みを初期化
        dummy_state = np.zeros(shape=(1, config.window_length) + config.env_observation_shape, dtype=np.float32)
        val = self(dummy_state)
        assert val.shape == (1, config.nb_actions)

    def call(self, state):
        return self.model(state)


# ------------------------------------------------------
# エピソード記憶部(episodic_reward)
# ------------------------------------------------------
class _EmbeddingNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        in_state, c = create_input_layers(
            config.window_length,
            config.env_observation_shape,
            config.env_observation_type,
            config.image_layer_type,
        )

        c = kl.Dense(
            32, activation="relu", kernel_initializer="he_normal", bias_initializer=keras.initializers.constant(0.001)
        )(c)
        self.model = keras.Model(in_state, c)

        # out layer
        self.concatrate = kl.Concatenate()
        self.d1 = kl.Dense(128, activation="relu", kernel_initializer="he_normal")
        c = kl.LayerNormalization()(c)
        self.out = kl.Dense(config.nb_actions, activation="softmax")

        # 重みを初期化
        dummy_state = np.zeros(shape=(1, config.window_length) + config.env_observation_shape, dtype=np.float32)
        val = self(dummy_state, dummy_state)
        assert val.shape == (1, config.nb_actions)

    def call(self, state1, state2):
        c1 = self.model(state1)
        c2 = self.model(state2)
        c = self.concatrate([c1, c2])
        c = self.d1(c)
        c = self.out(c)
        return c

    def predict(self, state):
        return self.model(state)


# ------------------------------------------------------
# 生涯記憶部(life long novelty module)
# ------------------------------------------------------
class _LifelongNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        in_state, c = create_input_layers(
            config.window_length,
            config.env_observation_shape,
            config.env_observation_type,
            config.image_layer_type,
        )

        c = kl.Dense(
            128,
            activation="relu",
            kernel_initializer="he_normal",
            bias_initializer="he_normal",
        )(c)
        c = kl.LayerNormalization()(c)
        self.model = keras.Model(in_state, c)

        # 重みを初期化
        dummy_state = np.zeros(shape=(1, config.window_length) + config.env_observation_shape, dtype=np.float32)
        val = self(dummy_state)
        assert val.shape == (1, 128)

    def call(self, state):
        return self.model(state)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.q_ext_online = _QNetwork(self.config)
        self.q_ext_target = _QNetwork(self.config)
        self.q_int_online = _QNetwork(self.config)
        self.q_int_target = _QNetwork(self.config)
        self.emb_network = _EmbeddingNetwork(self.config)
        self.lifelong_target = _LifelongNetwork(self.config)
        self.lifelong_train = _LifelongNetwork(self.config)

    def restore(self, data: Any) -> None:
        self.q_ext_online.set_weights(data[0])
        self.q_ext_target.set_weights(data[0])
        self.q_int_online.set_weights(data[1])
        self.q_int_target.set_weights(data[1])
        self.emb_network.set_weights(data[2])
        self.lifelong_target.set_weights(data[3])
        self.lifelong_train.set_weights(data[4])

    def backup(self):
        d = [
            self.q_ext_online.get_weights(),
            self.q_int_online.get_weights(),
            self.emb_network.get_weights(),
            self.lifelong_target.get_weights(),
            self.lifelong_train.get_weights(),
        ]
        return d

    def summary(self):
        self.q_ext_online.model.summary()
        self.emb_network.summary()
        self.lifelong_target.model.summary()

    # ---------------------------------

    def calc_target_q(
        self,
        q_online,
        q_target,
        rewards,
        n_states,
        done_list,
        gamma_list,
    ):
        n_q = q_online(n_states).numpy()
        n_q_target = q_target(n_states).numpy()

        target_q = []
        for i in range(len(rewards)):
            reward = rewards[i]
            if done_list[i]:
                gain = reward
            else:
                # DoubleDQN: indexはonlineQから選び、値はtargetQを選ぶ
                if self.config.enable_double_dqn:
                    n_act_idx = np.argmax(n_q[i])
                else:
                    n_act_idx = np.argmax(n_q_target[i])
                maxq = n_q_target[i][n_act_idx]
                gain = reward + gamma_list[i] * inverse_rescaling(maxq)
            gain = rescaling(gain)
            target_q.append(gain)
        target_q = np.asarray(target_q)

        return target_q


# ------------------------------------------------------
# RemoteMemory
# ------------------------------------------------------
class RemoteMemory(RLRemoteMemory):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.memory = factory.create(
            self.config.memory_name,
            {
                "capacity": self.config.capacity,
                "alpha": self.config.memory_alpha,
                "beta_initial": self.config.memory_beta_initial,
                "beta_steps": self.config.memory_beta_steps,
            },
        )
        self.invalid_memory = collections.deque(maxlen=self.config.capacity)

    def length(self) -> int:
        return len(self.memory)

    def restore(self, data: Any) -> None:
        self.memory.restore(data[0])
        self.invalid_memory = data[1]

    def backup(self):
        d = [self.memory.backup(), self.invalid_memory]
        return d

    # ---------------------------

    def add(self, batch, priority):
        self.memory.add(batch, priority)

    def sample(self, step: int) -> Tuple[list, list, list]:
        return self.memory.sample(self.config.batch_size, step)

    def update(self, indexes: List[int], batchs: List[Any], priorities: List[float]) -> None:
        self.memory.update(indexes, batchs, priorities)

    def length_invalid(self) -> int:
        return len(self.invalid_memory)

    def add_invalid(self, batch):
        self.invalid_memory.append(batch)

    def sample_invalid(self):
        return random.sample(self.invalid_memory, self.config.batch_size)

    def clear_invalid(self):
        return self.invalid_memory.clear()


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.memory = cast(RemoteMemory, self.memory)

        self.q_ext_optimizer = keras.optimizers.Adam(learning_rate=self.config.q_ext_lr)
        self.q_int_optimizer = keras.optimizers.Adam(learning_rate=self.config.q_int_lr)
        self.q_loss = keras.losses.Huber()

        self.emb_optimizer = keras.optimizers.Adam(learning_rate=self.config.episodic_lr)
        self.emb_loss = keras.losses.MeanSquaredError()

        self.lifelong_optimizer = keras.optimizers.Adam(learning_rate=self.config.lifelong_lr)
        self.lifelong_loss = keras.losses.MeanSquaredError()

        self.beta_list = create_beta_list(self.config.actor_num)
        self.gamma_list = create_gamma_list(self.config.actor_num)
        self.epsilon_list = create_epsilon_list(self.config.actor_num)

        self.train_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self):

        if self.memory.length() < self.config.memory_warmup_size:
            return {}

        indexes, batchs, weights = self.memory.sample(self.train_count)
        td_error, info = self._train_on_batchs(batchs, weights)
        priorities = abs(td_error) + 0.0001
        self.memory.update(indexes, batchs, priorities)

        # invalid action
        mem_invalid_len = self.memory.length_invalid()
        if mem_invalid_len > self.config.memory_warmup_size:
            batchs = self.memory.sample_invalid()
            self._train_on_batchs(batchs, [1 for _ in range(self.config.batch_size)])

        # targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_ext_target.set_weights(self.parameter.q_ext_online.get_weights())
            self.parameter.q_int_target.set_weights(self.parameter.q_int_online.get_weights())

        self.train_count += 1
        info["mem2"] = mem_invalid_len
        info["priority"] = np.mean(priorities)
        return info

    def _train_on_batchs(self, batchs, weights):

        # データ形式を変形
        states = []
        actions = []
        n_states = []
        rewards_ext = []
        rewards_int = []
        done_list = []
        gamma_list = []
        beta_list = []
        for b in batchs:
            states.append(b["states"][:-1])
            actions.append(b["action"])
            n_states.append(b["states"][1:])
            rewards_ext.append(b["reward_ext"])
            rewards_int.append(b["reward_int"])
            done_list.append(b["done"])
            gamma_list.append(self.gamma_list[b["actor"]])
            beta_list.append(self.beta_list[b["actor"]])
        states = np.asarray(states)
        n_states = np.asarray(n_states)
        rewards_ext = np.asarray(rewards_ext)
        rewards_int = np.asarray(rewards_int)

        actions_onehot = tf.one_hot(actions, self.config.nb_actions)

        # ----------------------------------------
        # Q network
        # ----------------------------------------
        _params = [
            states,
            n_states,
            actions_onehot,
            done_list,
            weights,
            gamma_list,
        ]
        td_error_ext, loss_ext = self._update_q(
            self.parameter.q_ext_online,
            self.parameter.q_ext_target,
            self.q_ext_optimizer,
            rewards_ext,
            *_params,
        )
        td_error_int, loss_int = self._update_q(
            self.parameter.q_int_online,
            self.parameter.q_int_target,
            self.q_int_optimizer,
            rewards_int,
            *_params,
        )

        # targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_ext_target.set_weights(self.parameter.q_ext_online.get_weights())
            self.parameter.q_int_target.set_weights(self.parameter.q_int_online.get_weights())

        # priority
        priorities = abs(td_error_ext + beta_list * td_error_int) + 0.0001

        # ----------------------------------------
        # embedding network
        # ----------------------------------------
        with tf.GradientTape() as tape:
            actions_prebs = self.parameter.emb_network(states, n_states)
            emb_loss = self.emb_loss(actions_prebs, actions_onehot)

        grads = tape.gradient(emb_loss, self.parameter.emb_network.trainable_variables)
        self.emb_optimizer.apply_gradients(zip(grads, self.parameter.emb_network.trainable_variables))

        # ----------------------------------------
        # lifelong network
        # ----------------------------------------
        lifelong_target_val = self.parameter.lifelong_target(states)
        with tf.GradientTape() as tape:
            lifelong_train_val = self.parameter.lifelong_train(states)
            lifelong_loss = self.lifelong_loss(lifelong_target_val, lifelong_train_val)

        grads = tape.gradient(lifelong_loss, self.parameter.lifelong_train.trainable_variables)
        self.lifelong_optimizer.apply_gradients(zip(grads, self.parameter.lifelong_train.trainable_variables))

        return priorities, {
            "loss_ext": loss_ext,
            "loss_int": loss_int,
            "emb_loss": emb_loss.numpy(),
            "lifelong_loss": lifelong_loss.numpy(),
        }

    def _update_q(
        self,
        q_online,
        q_target,
        optimizer,
        rewards,
        #
        states,
        n_states,
        actions_onehot,
        done_list,
        weights,
        gamma_list,
    ):
        target_q = self.parameter.calc_target_q(
            q_online,
            q_target,
            rewards,
            n_states,
            done_list,
            gamma_list,
        )

        with tf.GradientTape() as tape:
            q = q_online(states)
            q = tf.reduce_sum(q * actions_onehot, axis=1)
            loss = self.q_loss(target_q * weights, q * weights)

        grads = tape.gradient(loss, q_online.trainable_variables)
        optimizer.apply_gradients(zip(grads, q_online.trainable_variables))

        td_error = (target_q - q).numpy()
        loss = loss.numpy()

        return td_error, loss


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.memory = cast(RemoteMemory, self.memory)

        self.dummy_state = np.full(self.config.env_observation_shape, self.config.dummy_state_val, dtype=np.float32)
        self.invalid_action_reward = -1

        # actor
        self.beta_list = create_beta_list(self.config.actor_num)
        self.epsilon_list = create_epsilon_list(self.config.actor_num)
        self.gamma_list = create_gamma_list(self.config.actor_num)

        # ucb
        self.actor_index = -1
        self.ucb_recent = []
        self.ucb_actors_count = [1 for _ in range(self.config.actor_num)]  # 1回は保証
        self.ucb_actors_reward = [0.0 for _ in range(self.config.actor_num)]

    def on_reset(self, state: np.ndarray, valid_actions: List[int], _) -> None:
        if self.training:
            # エピソード毎に actor を決める
            self.actor_index = self._calc_actor_index()
            self.beta = self.beta_list[self.actor_index]
            self.epsilon = self.epsilon_list[self.actor_index]
            self.gamma = self.gamma_list[self.actor_index]

        else:
            self.epsilon = self.config.test_epsilon
            self.beta = self.config.test_beta

        self.recent_states = [self.dummy_state for _ in range(self.config.window_length + 1)]
        self.recent_states.pop(0)
        self.recent_states.append(state)

        # Q値取得用
        self.onehot_actor_idx = tf.one_hot(np.array(self.actor_index), self.config.actor_num)[np.newaxis, ...]

        # sliding-window UCB 用に報酬を保存
        self.episode_reward = 0.0

        # エピソードメモリ(エピソード毎に初期化)
        self.episodic_memory = collections.deque(maxlen=self.config.episodic_memory_capacity)

    # (sliding-window UCB)
    def _calc_actor_index(self) -> int:

        # UCB計算用に保存
        if self.actor_index != -1:
            self.ucb_recent.append(
                (
                    self.actor_index,
                    self.episode_reward,
                )
            )
            self.ucb_actors_count[self.actor_index] += 1
            self.ucb_actors_reward[self.actor_index] += self.episode_reward
            if len(self.ucb_recent) >= self.config.ucb_window_size:
                d = self.ucb_recent.pop(0)
                self.ucb_actors_count[d[0]] -= 1
                self.ucb_actors_reward[d[0]] -= d[1]

        N = len(self.ucb_recent)

        # 全て１回は実行
        if N < self.config.actor_num:
            return N

        # ランダムでactorを決定
        if random.random() < self.config.ucb_epsilon:
            return random.randint(0, self.config.actor_num - 1)

        # UCB値を計算
        ucbs = []
        for i in range(self.config.actor_num):
            n = self.ucb_actors_count[i]
            u = self.ucb_actors_reward[i] / n
            ucb = u + self.config.ucb_beta * np.sqrt(np.log(N) / n)
            ucbs.append(ucb)

        # UCB値最大のポリシー（複数あればランダム）
        return random.choice(np.where(ucbs == np.max(ucbs))[0])

    def policy(self, state_: np.ndarray, valid_actions: List[int], _):
        state = np.asarray([self.recent_states[1:]])

        q_ext = self.parameter.q_ext_online(state)[0].numpy()
        q_int = self.parameter.q_int_online(state)[0].numpy()
        q = q_ext + self.beta * q_int

        if random.random() < self.epsilon:
            action = random.choice(valid_actions)
        else:
            # valid actions以外は -inf にする
            q = np.array([(v if i in valid_actions else -np.inf) for i, v in enumerate(q)])
            # 最大値を選ぶ（複数はほぼないので無視）
            action = int(np.argmax(q))

        return action, (action, q[action])

    def on_step(
        self,
        state: np.ndarray,
        action_: Any,
        next_state: np.ndarray,
        reward_ext: float,
        done: bool,
        valid_actions: List[int],
        next_valid_actions: List[int],
        _,
    ):
        self.episode_reward += reward_ext

        self.recent_states.pop(0)
        self.recent_states.append(next_state)

        action = action_[0]
        q = action_[1]

        # 内部報酬
        n_s = np.asarray([self.recent_states[1:]])
        episodic_reward = self._calc_episodic_reward(n_s)
        lifelong_reward = self._calc_lifelong_reward(n_s)
        reward_int = episodic_reward * lifelong_reward

        _info = {
            "episodic": episodic_reward,
            "lifelong": lifelong_reward,
            "reward_int": reward_int,
        }
        if not self.training:
            return _info

        batch = {
            "states": self.recent_states[:],
            "action": action,
            "reward_ext": reward_ext,
            "reward_int": reward_int,
            "done": done,
            "actor": self.actor_index,
        }

        # priority
        if self.config.memory_name == "ReplayMemory":
            priority = 1
        else:
            _params = [
                n_s,
                [done],
                [self.gamma],
            ]
            target_q_ext = self.parameter.calc_target_q(
                self.parameter.q_ext_online,
                self.parameter.q_ext_target,
                [reward_ext],
                *_params,
            )
            target_q_int = self.parameter.calc_target_q(
                self.parameter.q_int_online,
                self.parameter.q_int_target,
                [reward_int],
                *_params,
            )
            priority = abs((target_q_ext + self.beta * target_q_int) - q) + 0.0001

        self.memory.add(batch, priority)

        # --- invalid action
        if self.invalid_action_reward > reward_ext - 1:
            self.invalid_action_reward = reward_ext - 1
            self.memory.clear_invalid()
        # TODO

        _info["priority"] = priority
        return _info

    def _calc_episodic_reward(self, state):
        k = self.config.episodic_count_max
        epsilon = self.config.episodic_epsilon
        cluster_distance = self.config.episodic_cluster_distance
        c = self.config.episodic_pseudo_counts

        # 埋め込み関数から制御可能状態を取得
        cont_state = self.parameter.emb_network.predict(state)[0].numpy()

        # 初回
        if len(self.episodic_memory) == 0:
            self.episodic_memory.append(cont_state)
            return 1 / c

        # エピソードメモリ内の全要素とユークリッド距離を求める
        euclidean_list = [np.linalg.norm(m - cont_state, ord=2) for m in self.episodic_memory]

        # エピソードメモリに制御可能状態を追加
        self.episodic_memory.append(cont_state)

        # 近いk個を対象
        euclidean_list = np.sort(euclidean_list)[:k]

        # 上位k個の移動平均を出す
        mode_ave = np.mean(euclidean_list)
        if mode_ave == 0.0:
            # ユークリッド距離は正なので平均0は全要素0のみ
            dn = euclidean_list
        else:
            dn = euclidean_list / mode_ave  # 正規化

        # 一定距離以下を同じ状態とする
        dn = np.maximum(dn - cluster_distance, 0)

        # 訪問回数を計算(Dirac delta function の近似)
        dn = epsilon / (dn + epsilon)
        N = np.sum(dn)

        # 報酬の計算
        reward = 1 / (np.sqrt(N) + c)
        return reward

    def _calc_lifelong_reward(self, state):
        # RND取得
        rnd_target_val = self.parameter.lifelong_target(state)[0]
        rnd_train_val = self.parameter.lifelong_train(state)[0]

        # MSE
        error = np.square(rnd_target_val - rnd_train_val).mean()
        reward = 1 + error

        if reward < 1:
            reward = 1
        if reward > self.config.lifelong_max:
            reward = self.config.lifelong_max

        return reward

    def render(
        self,
        state: np.ndarray,
        valid_actions: List[int],
        action_to_str,
    ) -> None:
        state = np.asarray([self.recent_states[1:]])
        q_ext = self.parameter.q_ext_online(state)[0].numpy()
        q_int = self.parameter.q_int_online(state)[0].numpy()
        q = q_ext + self.beta * q_int

        maxa = np.argmax(q)
        for a in range(self.config.nb_actions):
            if a not in valid_actions:
                s = "x"
            else:
                s = " "
            if a == maxa:
                s += "*"
            else:
                s += " "
            s += f"{action_to_str(a)}: {q[a]:5.3f} = {q_ext[a]:5.3f} + {self.beta} * {q_int[a]:5.3f}"
            print(s)


if __name__ == "__main__":
    pass

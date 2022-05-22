import pickle
import random
from dataclasses import dataclass
from typing import Any, Tuple, cast

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from srl.base.define import RLObservationType
from srl.base.env.base import EnvBase
from srl.base.rl.algorithms.discrete_action import DiscreteActionConfig
from srl.base.rl.base import RLParameter, RLTrainer, RLWorker
from srl.base.rl.registration import register
from srl.base.rl.remote_memory.sequence_memory import SequenceRemoteMemory
from srl.rl.functions.common import random_choice_by_probs, render_discrete_action, to_str_observation
from srl.rl.functions.model import ImageLayerType, create_input_layers_one_sequence
from tensorflow.keras import layers as kl


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(DiscreteActionConfig):

    simulation_times: int = 100
    action_select_threshold: int = 10
    gamma: float = 1.0  # 割引率

    puct_c: float = 1.0
    early_steps: int = 0
    lr: float = 0.001
    batch_size: int = 16
    train_size: int = 1_000
    epochs: int = 5

    # model
    window_length: int = 1
    hidden_layer_sizes: Tuple[int, ...] = (512,)
    activation: str = "relu"
    image_layer_type: ImageLayerType = ImageLayerType.AlphaZero

    def __post_init__(self):
        super().__init__()

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.CONTINUOUS

    @staticmethod
    def getName() -> str:
        return "AlphaZero"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.batch_size < self.train_size


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
class RemoteMemory(SequenceRemoteMemory):
    pass


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _Network(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        in_state, c = create_input_layers_one_sequence(
            # config.window_length,
            config.observation_shape,
            config.env_observation_type,
            config.image_layer_type,
        )

        # --- hidden layer
        for h in config.hidden_layer_sizes:
            c = kl.Dense(h, activation=config.activation, kernel_initializer="he_normal")(c)

        # --- out layer
        policy = kl.Dense(config.nb_actions, activation="softmax", bias_initializer="truncated_normal")(c)
        value = kl.Dense(1, bias_initializer="truncated_normal")(c)
        self.model = keras.Model(in_state, [policy, value])

        # 重みを初期化
        dummy_state = np.zeros(shape=(1,) + config.observation_shape, dtype=np.float32)
        # dummy_state = np.zeros(shape=(1, config.window_length) + config.env_observation_shape, dtype=np.float32)
        policy, value = self(dummy_state)
        assert policy.shape == (1, config.nb_actions)
        assert value.shape == (1, 1)

    def call(self, state):
        return self.model(state)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.network = _Network(self.config)

        self.N = {}  # 訪問回数
        self.W = {}  # 累計報酬

        # cache (simulationで繰り返すので)
        self.reset_cache()

    def restore(self, data: Any) -> None:
        d = pickle.loads(data)
        self.N = d[0]
        self.W = d[1]
        self.network.set_weights(d[2])
        self.reset_cache()

    def backup(self):
        return pickle.dumps(
            [
                self.N,
                self.W,
                self.network.get_weights(),
            ]
        )

    def summary(self):
        self.network.model.summary()

    # ------------------------

    def init_state(self, state_str):
        if state_str not in self.N:
            self.W[state_str] = [0 for _ in range(self.config.nb_actions)]
            self.N[state_str] = [0 for _ in range(self.config.nb_actions)]

    def pred_PV(self, state, state_str):
        if state_str not in self.P:
            p, v = self.network(np.asarray([state]))
            self.P[state_str] = p[0].numpy()
            self.V[state_str] = v[0][0].numpy()

    def reset_cache(self):
        self.P = {}
        self.V = {}


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
        self.value_loss = keras.losses.MeanSquaredError()
        self.policy_loss = keras.losses.CategoricalCrossentropy()

        self.train_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self):

        if self.remote_memory.length() < self.config.train_size:
            return {}
        batchs = self.remote_memory.sample()
        states = []
        policies = []
        rewards = []
        for b in batchs:
            states.append(b["state"])
            policies.append(b["policy"])
            rewards.append(b["reward"])
        states = np.asarray(states)
        policies = np.asarray(policies)
        rewards = np.asarray(rewards).reshape((-1, 1))

        indexes = [i for i in range(len(batchs))]
        for _ in range(self.config.epochs):
            idx = random.sample(indexes, self.config.batch_size)
            state = states[idx]
            policy = policies[idx]
            reward = rewards[idx]

            with tf.GradientTape() as tape:
                p_pred, v_pred = self.parameter.network(state)

                # value: 状態に対する勝率(reward)を教師に学習
                value_loss = self.value_loss(reward, v_pred)

                # policy: 選んだアクション(MCTSの結果)を教師に学習
                policy_loss = self.policy_loss(policy, p_pred)

                loss = tf.reduce_mean(value_loss + policy_loss)

            grads = tape.gradient(loss, self.parameter.network.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.parameter.network.trainable_variables))

            self.train_count += 1

        # 学習したらキャッシュは削除
        self.parameter.reset_cache()

        return {
            "value_loss": value_loss.numpy(),
            "policy_loss": policy_loss.numpy(),
        }


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

    def _on_reset(self, state: np.ndarray, player_index: int, env: EnvBase) -> None:
        self.step = 0
        self.state = state
        self.state_str = to_str_observation(state)
        self.invalid_actions = env.get_invalid_actions(player_index)

        self.history = []

    def _policy(self, state: np.ndarray, player_index: int, env: EnvBase) -> int:

        state_str = to_str_observation(state)
        self.parameter.N = {}
        self.parameter.W = {}
        self.parameter.init_state(state_str)

        # シミュレーションしてpolicyを作成
        dat = env.backup()
        for _ in range(self.config.simulation_times):
            self._simulation(state, state_str, env, player_index)
            env.restore(dat)
        N = sum(self.parameter.N[state_str])
        n = self.parameter.N[state_str]
        policy = [n[a] / N for a in range(self.config.nb_actions)]

        if self.step < self.config.early_steps:
            # episodeの序盤は試行回数に比例した確率でアクションを選択
            probs = np.array([0 if a in self.invalid_actions else v for a, v in enumerate(policy)])  # mask
            action = random_choice_by_probs(probs)
        else:
            # 確率が一番高いアクションを採用
            action = random.choice(np.where(policy == np.max(policy))[0])

        # self.policy_ = [1 if action == a else 0 for a in range(self.config.nb_actions)]
        self.policy_ = policy

        self.remote_memory.add(
            {
                "state": state,
                "policy": self.policy_,
                "reward": self.parameter.W[state_str][action] / self.parameter.N[state_str][action],
            }
        )

        return int(action)

    def _simulation(self, state: np.ndarray, state_str: str, env: EnvBase, player_index, depth: int = 0):
        if depth >= env.max_episode_steps:  # for safety
            return 0

        action = self._select_action(env, state, state_str, player_index)

        # ロールアウト
        if False:
            # if self.parameter.N[state_str][action] < self.config.action_select_threshold:
            reward = self.parameter.V[state_str]
        else:
            next_player_indices = env.get_next_player_indices()
            assert player_index in next_player_indices

            # --- steps
            reward = 0
            n_state = state
            n_state_str = state_str
            while True:
                actions = [self._select_action(env, n_state, n_state_str, idx) for idx in next_player_indices]
                n_state, rewards, done, next_player_indices, _ = env.step(actions)
                n_state = self.observation_encode(n_state, env)
                n_state_str = to_str_observation(n_state)
                reward += rewards[player_index]

                if player_index in next_player_indices:
                    break
                if done:
                    break

            if done:
                pass  # 終了
            else:
                # 展開
                reward += self._simulation(n_state, n_state_str, env, player_index, depth + 1)

        # 結果を記録
        self.parameter.N[state_str][action] += 1
        self.parameter.W[state_str][action] += reward
        # self.remote_memory.add_simulation  TODO remote

        return reward * self.config.gamma  # 割り引いて前に伝搬

    def _select_action(self, env, state, state_str, idx):
        self.parameter.init_state(state_str)
        self.parameter.pred_PV(state, state_str)

        invalid_actions = env.get_invalid_actions(idx)
        scores = self._calc_puct(state_str, invalid_actions)

        action = random.choice(np.where(scores == np.max(scores))[0])
        return int(action)

    def _calc_puct(self, state_str, invalid_actions):
        # --- PUCTに従ってアクションを選択
        N = np.sum(self.parameter.N[state_str])
        scores = []
        for a in range(self.config.nb_actions):
            if a in invalid_actions:
                score = -np.inf
            else:
                # P(s,a): 過去のMCTSの結果を教師あり学習した結果
                # U(s,a) = C_puct * P(s,a) * sqrt(ΣN(s)) / (1+N(s,a))
                # score = Q(s,a) + U(s,a)
                P = self.parameter.P[state_str][a]
                n = self.parameter.N[state_str][a]
                u = self.config.puct_c * P * (np.sqrt(N) / (1 + n))
                if self.parameter.N[state_str][a] == 0:
                    q = 0
                else:
                    q = self.parameter.W[state_str][a] / n
                score = q + u
            scores.append(score)
        return scores

    def _on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        player_index: int,
        env: EnvBase,
    ):
        self.history.append([self.state, self.policy_, reward])

        self.step += 1
        self.state = next_state
        self.state_str = to_str_observation(next_state)
        self.invalid_actions = env.get_invalid_actions(player_index)

        if False and done and self.training:
            # 報酬を逆伝搬
            reward = 0
            for h in reversed(self.history):
                reward = h[2] + self.config.gamma * reward
                self.remote_memory.add(
                    {
                        "state": h[0],
                        "policy": h[1],
                        "reward": reward,
                    }
                )

        return {}

    def render(self, env: EnvBase, player_index: int) -> None:
        self.parameter.init_state(self.state_str)
        self.parameter.pred_PV(self.state, self.state_str)
        print(f"value: {self.parameter.V[self.state_str]:7.3f}")
        puct = self._calc_puct(self.state_str, self.invalid_actions)
        invalid_actions = self.get_invalid_actions(env, player_index)

        def _render_sub(a: int) -> str:
            if self.state_str in self.parameter.W:
                q = self.parameter.W[self.state_str][a]
                c = self.parameter.N[self.state_str][a]
                if c != 0:
                    q /= c
            else:
                q = 0
                c = 0
            s = "{}: {:9.5f}({:7d}), policy {:9.5f}, puct {:.5f}".format(
                env.action_to_str(a),
                q,
                c,
                self.parameter.P[self.state_str][a],
                puct[a],
            )
            return s

        render_discrete_action(invalid_actions, None, env, _render_sub)

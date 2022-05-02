import collections
import json
import random
from dataclasses import dataclass
from typing import Any, List, Tuple, cast

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from srl.base.rl import DiscreteActionConfig, RLParameter, RLRemoteMemory, RLTrainer, RLWorker
from srl.base.rl.env_for_rl import EnvForRL
from srl.base.rl.registory import register
from srl.rl.functions.common import to_str_observaten
from srl.rl.functions.model import ImageLayerType, create_input_layers, create_input_layers_one_sequence
from tensorflow.keras import layers as kl


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(DiscreteActionConfig):

    simulation_times: int = 1
    action_select_threshold: int = 10
    gamma: float = 1.0  # 割引率

    puct_c: float = 1.0
    early_steps: int = 0
    lr: float = 0.01
    batch_size: int = 16
    buffer_size: int = 100_000
    warmup_size: int = 100

    # model
    window_length: int = 1
    hidden_layer_sizes: Tuple[int, ...] = (512,)
    activation: str = "relu"
    image_layer_type: ImageLayerType = ImageLayerType.DQN

    @staticmethod
    def getName() -> str:
        return "AlphaZero"


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
class RemoteMemory(RLRemoteMemory):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.buffer_sim = []
        self.buffer_policy = collections.deque(maxlen=self.config.buffer_size)

    def length(self) -> int:
        return len(self.buffer_policy)

    def restore(self, data: Any) -> None:
        self.buffer_sim = data[0]
        self.buffer_policy = data[1]

    def backup(self):
        return [
            self.buffer_sim,
            self.buffer_policy,
        ]

    # ------------------------
    def add_sim(self, batch: Any) -> None:
        self.buffer_sim.append(batch)

    def get_sim(self):
        buffer = self.buffer_sim
        self.buffer_sim = []
        return buffer

    def add_policy(self, batch: Any) -> None:
        self.buffer_policy.append(batch)

    def sample(self):
        return random.sample(self.buffer_policy, self.config.batch_size)


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _Network(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        in_state, c = create_input_layers_one_sequence(
            # config.window_length,
            config.env_observation_shape,
            config.env_observation_type,
            config.image_layer_type,
        )

        # --- hidden layer
        for h in config.hidden_layer_sizes:
            c = kl.Dense(h, activation=config.activation, kernel_initializer="he_normal")(c)

        # --- out layer
        policy = kl.Dense(config.nb_actions, activation="softmax")(c)
        value = kl.Dense(1, activation="tanh")(c)
        self.model = keras.Model(in_state, [policy, value])

        # 重みを初期化
        dummy_state = np.zeros(shape=(1,) + config.env_observation_shape, dtype=np.float32)
        # dummy_state = np.zeros(shape=(1, config.window_length) + config.env_observation_shape, dtype=np.float32)
        policy, value = self(dummy_state)
        assert policy.shape == (1, config.nb_actions)
        assert value.shape == (1, 1)

    def call(self, state):
        policy, value = self.model(state)
        return policy, value


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
        self.P = {}
        self.V = {}

    def restore(self, data: Any) -> None:
        d = json.loads(data)
        self.N = d[0]
        self.W = d[1]
        self.network.set_weights(d[2])
        self.P = d[3]
        self.V = d[4]

    def backup(self):
        return json.dumps(
            [
                self.N,
                self.W,
                self.network.get_weights(),
                self.P,
                self.V,
            ]
        )

    # ------------------------

    def init_state(self, state_str):
        if state_str not in self.N:
            self.W[state_str] = [0 for _ in range(self.config.nb_actions)]
            self.N[state_str] = [0 for _ in range(self.config.nb_actions)]

    def set_cache(self, state, state_str):
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
        # シミュレーションがアクション前ではなくアクション後になっている点が異なっています

        info = {}

        batchs = self.memory.get_sim()
        for batch in batchs:
            state_str = batch["state_str"]
            action = batch["action"]
            reward = batch["reward"]

            self.parameter.init_state(state_str)
            self.parameter.N[state_str][action] += 1
            self.parameter.W[state_str][action] += reward

        if self.memory.length() > self.config.warmup_size:

            batchs = self.memory.sample()

            values = []
            states = []
            policies = []
            for b in batchs:
                states.append(b["state"])
                policies.append(b["policy"])
                state_str = str(b["state"].tolist())
                self.parameter.init_state(state_str)

                # 状態価値(policyは均等)
                value = 0
                for a in range(self.config.nb_actions):
                    if self.parameter.N[state_str][a] > 0:
                        value += self.parameter.W[state_str][a] / self.parameter.N[state_str][a]
                values.append(value / self.config.nb_actions)

            states = np.asarray(states)
            values = np.asarray(values).reshape((-1, 1))
            policies = np.asarray(policies)

            with tf.GradientTape() as tape:
                p_pred, v_pred = self.parameter.network(states)

                # value: 状態に対する勝率(reward)を教師に学習
                value_loss = self.value_loss(values, v_pred)

                # policy: 選んだアクション(MCTSの結果)を教師に学習
                policy_loss = self.policy_loss(policies, p_pred)

                # loss
                loss = tf.reduce_mean(value_loss + policy_loss)

            grads = tape.gradient(loss, self.parameter.network.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.parameter.network.trainable_variables))

            # 学習したらキャッシュは削除
            self.parameter.reset_cache()
            self.train_count += 1

            info["value_loss"] = value_loss.numpy()
            info["policy_loss"] = policy_loss.numpy()

        return info

    def _simulation(self, env: EnvForRL, state, valid_actions, depth: int = 0):
        if depth >= env.max_episode_steps:  # for safety
            return 0

        s = str(state.tolist())
        self.parameter.init_state(s)
        self.parameter.set_cache(state, s)

        # --- PUCTに従ってアクションを選択
        N = np.sum(self.parameter.N[s])
        scores = []
        for a in range(self.config.nb_actions):
            if a not in valid_actions:
                score = -np.inf
            else:
                # P(s,a): 過去のMCTSの結果を教師あり学習した結果
                # U(s,a) = C_puct * P(s,a) * sqrt(ΣN(s)) / (1+N(s,a))
                puct = self.config.puct_c * self.parameter.P[s][a] * (np.sqrt(N) / (1 + self.parameter.N[s][a]))
                # score = Q(s,a) + U(s,a)
                if self.parameter.N[s][a] == 0:
                    q = 0
                else:
                    q = self.parameter.W[s][a] / self.parameter.N[s][a]
                score = q + puct
            scores.append(score)
        action = random.choice(np.where(scores == np.max(scores))[0])

        # --- step
        if self.parameter.N[s][action] < self.config.action_select_threshold:
            # ロールアウトの代わりにNNの結果を返す
            reward = self.parameter.V[s]
        else:
            # step
            n_state, reward, done, _ = env.step(action)
            n_valid_actions = env.fetch_valid_actions()

            if done:
                pass  # 終了(終了時の報酬が結果)
            else:
                # 展開
                reward += self._simulation(env, n_state, n_valid_actions, depth + 1)

        # 結果を記録
        batch = {
            "state_str": s,
            "action": action,
            "reward": reward,
        }
        self.memory.add_sim(batch)

        return reward * self.config.gamma  # 割り引いて前に伝搬


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

    def on_reset(self, state: np.ndarray, player_index: int, env: EnvForRL) -> None:
        self.step = 0
        self.state = to_str_observaten(state)
        self.invalid_actions = env.get_invalid_actions(player_index)

    def policy(self, state: np.ndarray, player_index: int, env: EnvForRL) -> int:

        if self.training:
            # シミュレーション
            for _ in range(self.config.simulation_times):
                save_state = env.backup()
                self._simulation(env, state, valid_actions)
                env.restore(save_state)

        if s in self.parameter.N:

            N = sum(self.parameter.N[s])
            n = self.parameter.N[s]

            # 0%を回避するために1回は保証
            policy = [(n[a] + 1) / (N + nb_actions) for a in range(nb_actions)]

            # episodeの序盤は試行回数に比例した確率でアクションを選択
            if self.step < self.config.early_steps:
                probs = np.array([v if a in valid_actions else 0 for a, v in enumerate(policy)])  # mask
                probs = probs / sum(probs)  # total 1
                action = np.random.choice(range(self.config.nb_actions), p=probs)

            # 試行回数のもっとも多いアクションを採用
            else:
                n = [n[a] if a in valid_actions else -np.inf for a in range(self.config.nb_actions)]  # mask
                action = random.choice(np.where(n == np.max(n))[0])
        else:
            action = random.choice(valid_actions)
            policy = [1 / nb_actions for _ in range(nb_actions)]

        self.memory.add_policy(
            {
                "state": state,
                "policy": policy,
            }
        )

        return action, policy

    def on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        player_index: int,
        env: EnvForRL,
    ):
        self.state = to_str_observaten(next_state)
        self.invalid_actions = env.get_invalid_actions(player_index)

        self.step += 1
        return {}

    def render(self, env: EnvForRL) -> None:
        self.parameter.init_state(self.state)
        self.parameter.set_cache(state, s)
        print(f"value: {self.parameter.V[s]:7.3f}")
        for a in range(self.config.nb_actions):
            if s in self.parameter.W:
                q = self.parameter.W[s][a]
                c = self.parameter.N[s][a]
                if c != 0:
                    q /= c
            else:
                q = 0
                c = 0
            print(f"{action_to_str(a)}: {q:7.3f}({c:7d}), policy {self.parameter.P[s][a]:7.3f}")


if __name__ == "__main__":
    pass

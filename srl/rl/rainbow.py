import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from srl.base.env.env_for_rl import EnvForRL
from srl.base.rl.algorithms.neuralnet_discrete import DiscreteActionConfig, DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.registration import register
from srl.rl.functions.common import calc_epsilon_greedy_probs, random_choice_by_probs
from srl.rl.functions.dueling_network import create_dueling_network_layers
from srl.rl.functions.model import ImageLayerType, create_input_layers
from srl.base.rl.remote_memory import PriorityExperienceReplay
from tensorflow.keras import layers as kl

"""
DQN
    window_length               : o (config selection)
    Target Network              : o
    Huber loss function         : o
    Delay update Target Network : o
    Experience Replay  : o
    Frame skip         : -
    Annealing e-greedy : o (config selection)
    Reward clip        : o (config selection)
    Image preprocessor : -
Rainbow
    Double DQN                  : o (config selection)
    Priority Experience Replay  : o (config selection)
    Dueling Network             : o (config selection)
    Multi-Step learning(retrace): o (config selection)
    Noisy Network               : o (config selection)
    Categorical DQN             : x
Other
    invalid_actions : o

"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(DiscreteActionConfig):

    test_epsilon: float = 0

    epsilon: float = 0.1
    # Annealing e-greedy
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.01
    exploration_steps: int = -1

    # model
    window_length: int = 1
    hidden_layer_sizes: Tuple[int, ...] = (512,)
    activation: str = "relu"
    image_layer_type: ImageLayerType = ImageLayerType.DQN
    enable_noisy_dense: bool = False

    gamma: float = 0.99  # 割引率
    lr: float = 0.001  # 学習率
    batch_size: int = 32
    target_model_update_interval: int = 100
    reward_clip: Optional[Tuple[float, float]] = None
    enable_double_dqn: bool = True
    multisteps: int = 3
    retrace_h: float = 1.0

    # DuelingNetwork
    enable_dueling_network: bool = True
    dueling_network_type: str = "average"

    # Priority Experience Replay
    capacity: int = 100_000
    memory_name: str = "RankBaseMemory"
    memory_warmup_size: int = 1000
    memory_alpha: float = 0.8
    memory_beta_initial: float = 0.4
    memory_beta_steps: int = 1_000_000

    dummy_state_val: float = 0.0

    @staticmethod
    def getName() -> str:
        return "Rainbow"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.window_length > 0
        assert self.memory_warmup_size < self.capacity
        assert self.batch_size < self.memory_warmup_size
        assert len(self.hidden_layer_sizes) > 0


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
class RemoteMemory(PriorityExperienceReplay):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.init(
            self.config.memory_name,
            self.config.capacity,
            self.config.memory_alpha,
            self.config.memory_beta_initial,
            self.config.memory_beta_steps,
        )


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

        if config.enable_noisy_dense:
            Dense = tfa.layers.NoisyDense
        else:
            Dense = kl.Dense

        for i in range(len(config.hidden_layer_sizes) - 1):
            c = Dense(
                config.hidden_layer_sizes[i],
                activation=config.activation,
                kernel_initializer="he_normal",
            )(c)

        if config.enable_dueling_network:
            c = create_dueling_network_layers(
                c,
                config.nb_actions,
                config.hidden_layer_sizes[-1],
                config.dueling_network_type,
                activation=config.activation,
                enable_noisy_dense=config.enable_noisy_dense,
            )
        else:
            c = Dense(config.hidden_layer_sizes[-1], activation=config.activation, kernel_initializer="he_normal")(c)
            c = Dense(config.nb_actions, kernel_initializer="truncated_normal", bias_initializer="truncated_normal")(c)

        self.model = keras.Model(in_state, c)

        # 重みを初期化
        in_shape = (config.window_length,) + config.env_observation_shape
        dummy_state = np.zeros(shape=(1,) + in_shape, dtype=np.float32)
        val = self(dummy_state)
        assert val.shape == (1, config.nb_actions)

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

    def summary(self):
        self.q_online.model.summary()

    # ----------------------------------------------

    def calc_target_q(self, batchs):

        n_states_list = []
        for b in batchs:
            n_states_list.extend(b["states"][1:])
        n_states_list = np.asarray(n_states_list)

        n_q_list = self.q_online(n_states_list).numpy()
        n_q_list_target = self.q_target(n_states_list).numpy()

        target_q_list = []
        n_states_idx_start = 0
        for b in batchs:
            target_q = 0.0
            retrace = 1.0
            n_states_idx = n_states_idx_start
            for n in range(len(b["rewards"])):

                action = b["actions"][n]
                mu_prob = b["probs"][n]
                reward = b["rewards"][n]
                invalid_actions = b["invalid_actions"][n]
                next_invalid_actions = b["invalid_actions"][n + 1]
                done = b["dones"][n]

                # retrace
                if n >= 1:
                    pi_probs = calc_epsilon_greedy_probs(
                        n_q_list[n_states_idx - 1],
                        invalid_actions,
                        0.0,
                        self.config.nb_actions,
                    )
                    retrace *= self.config.retrace_h * np.minimum(1, pi_probs[action] / mu_prob)
                    if retrace == 0:
                        break  # 0以降は伝搬しないので切りあげる

                if done:
                    gain = reward
                else:
                    # DoubleDQN: indexはonlineQから選び、値はtargetQを選ぶ
                    if self.config.enable_double_dqn:
                        n_pi_probs = calc_epsilon_greedy_probs(
                            n_q_list[n_states_idx],
                            next_invalid_actions,
                            0.0,
                            self.config.nb_actions,
                        )
                    else:
                        n_pi_probs = calc_epsilon_greedy_probs(
                            n_q_list_target[n_states_idx],
                            next_invalid_actions,
                            0.0,
                            self.config.nb_actions,
                        )
                    P = 0
                    for j in range(self.config.nb_actions):
                        P += n_pi_probs[j] * n_q_list_target[n_states_idx][j]
                    gain = reward + self.config.gamma * P

                if n == 0:
                    target_q += gain
                else:
                    td_error = gain - n_q_list[n_states_idx - 1][action]
                    target_q += (self.config.gamma**n) * retrace * td_error
                n_states_idx += 1
            n_states_idx_start += len(b["rewards"])
            target_q_list.append(target_q)

        return np.asarray(target_q_list)


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

    def get_train_count(self):
        return self.train_count

    def train(self):

        if self.remote_memory.length() < self.config.memory_warmup_size:
            return {}

        indices, batchs, weights = self.remote_memory.sample(self.train_count, self.config.batch_size)
        td_error, loss = self._train_on_batchs(batchs, weights)
        priorities = abs(td_error) + 0.0001
        self.remote_memory.update(indices, batchs, priorities)

        # targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.set_weights(self.parameter.q_online.get_weights())

        self.train_count += 1
        return {
            "loss": loss,
            "priority": np.mean(priorities),
        }

    def _train_on_batchs(self, batchs, weights):

        states = []
        actions = []
        for b in batchs:
            states.append(b["states"][0])
            actions.append(b["actions"][0])
        states = np.asarray(states)

        target_q_list = self.parameter.calc_target_q(batchs)

        with tf.GradientTape() as tape:
            q = self.parameter.q_online(states)

            actions_onehot = tf.one_hot(actions, self.config.nb_actions)
            q = tf.reduce_sum(q * actions_onehot, axis=1)

            loss = self.loss(target_q_list * weights, q * weights)

        grads = tape.gradient(loss, self.parameter.q_online.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.q_online.trainable_variables))

        td_error = (target_q_list - q).numpy()
        return td_error, loss.numpy()


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(DiscreteActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

        self.dummy_state = np.full(self.config.env_observation_shape, self.config.dummy_state_val, dtype=np.float32)
        self.epsilon_step = 0

        if self.config.exploration_steps > 0:
            self.initial_epsilon = self.config.initial_epsilon
            self.epsilon_step = (
                self.config.initial_epsilon - self.config.final_epsilon
            ) / self.config.exploration_steps
            self.final_epsilon = self.config.final_epsilon

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> None:
        self.recent_states = [self.dummy_state for _ in range(self.config.window_length)]
        self.recent_bundle_states = [self.recent_states[:] for _ in range(self.config.multisteps + 1)]

        self.recent_actions = [random.randint(0, self.config.nb_actions - 1) for _ in range(self.config.multisteps)]
        self.recent_probs = [1.0 / self.config.nb_actions for _ in range(self.config.multisteps)]
        self.recent_rewards = [0.0 for _ in range(self.config.multisteps)]
        self.recent_done = [False for _ in range(self.config.multisteps)]
        self.recent_invalid_actions = [[] for _ in range(self.config.multisteps + 1)]

        self.recent_states.pop(0)
        self.recent_states.append(state)
        self.recent_bundle_states.pop(0)
        self.recent_bundle_states.append(self.recent_states[:])
        self.recent_invalid_actions.pop(0)
        self.recent_invalid_actions.append(invalid_actions)

    def call_policy(self, _state: np.ndarray, invalid_actions: List[int]) -> int:
        state = self.recent_bundle_states[-1]
        q = self.parameter.q_online(np.asarray([state]))[0].numpy()

        if self.config.enable_noisy_dense:
            action = int(np.argmax(q))
            self.action = action
            self.prob = 1.0
            self.q = q[action]
            return action

        if self.training:
            if self.config.exploration_steps > 0:
                # Annealing ε-greedy
                epsilon = self.initial_epsilon - self.epsilon_step * self.epsilon_step
                if epsilon < self.final_epsilon:
                    epsilon = self.final_epsilon
            else:
                epsilon = self.config.epsilon
        else:
            epsilon = self.config.test_epsilon

        probs = calc_epsilon_greedy_probs(q, invalid_actions, epsilon, self.config.nb_actions)
        self.action = random_choice_by_probs(probs)

        self.prob = probs[self.action]
        self.q = q[self.action]
        return self.action

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
    ):
        self.recent_states.pop(0)
        self.recent_states.append(next_state)
        self.recent_bundle_states.pop(0)
        self.recent_bundle_states.append(self.recent_states[:])
        self.recent_invalid_actions.pop(0)
        self.recent_invalid_actions.append(next_invalid_actions)

        if not self.training:
            return {}
        self.epsilon_step += 1

        # reward clip
        if self.config.reward_clip is not None:
            if reward < self.config.reward_clip[0]:
                reward = self.config.reward_clip[0]
            elif reward > self.config.reward_clip[1]:
                reward = self.config.reward_clip[1]

        self.recent_actions.pop(0)
        self.recent_actions.append(self.action)
        self.recent_probs.pop(0)
        self.recent_probs.append(self.prob)
        self.recent_rewards.pop(0)
        self.recent_rewards.append(reward)
        self.recent_done.pop(0)
        self.recent_done.append(done)

        priority = self._add_memory(self.q, None)

        if done:
            # 残りstepも追加
            for _ in range(len(self.recent_rewards) - 1):
                self.recent_bundle_states.pop(0)
                self.recent_actions.pop(0)
                self.recent_probs.pop(0)
                self.recent_rewards.pop(0)
                self.recent_done.pop(0)
                self.recent_invalid_actions.pop(0)

                self._add_memory(self.q, priority)

        return {"priority": priority}

    def _add_memory(self, q, priority):

        batch = {
            "states": self.recent_bundle_states[:],
            "actions": self.recent_actions[:],
            "probs": self.recent_probs[:],
            "rewards": self.recent_rewards[:],
            "dones": self.recent_done[:],
            "invalid_actions": self.recent_invalid_actions[:],
        }

        # priority
        if priority is None:
            if self.config.memory_name == "ReplayMemory":
                priority = 1
            else:
                target_q = self.parameter.calc_target_q([batch])[0]
                priority = abs(target_q - q) + 0.0001

        self.remote_memory.add(batch, priority)
        return priority

    def render(self, env: EnvForRL) -> None:
        invalid_actions = self.recent_invalid_actions[-1]

        state = self.recent_bundle_states[-1]
        q = self.parameter.q_online(np.asarray([state]))[0].numpy()
        maxa = np.argmax(q)
        for a in range(self.config.nb_actions):
            if len(invalid_actions) > 10:
                if a in invalid_actions:
                    continue
                s = ""
            else:
                if a in invalid_actions:
                    s = "x"
                else:
                    s = " "
            if a == maxa:
                s += "*"
            else:
                s += " "
            s += f"{env.action_to_str(a)}: {q[a]:.7f}"
            print(s)


if __name__ == "__main__":
    pass

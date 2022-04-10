import random
from collections import deque
from dataclasses import dataclass
from typing import Any, List, Tuple, cast

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from srl.base.rl import DiscreteActionConfig, RLParameter, RLRemoteMemory, RLTrainer, RLWorker
from srl.rl.functions.common import calc_epsilon_greedy_probs, inverse_rescaling, random_choice_by_probs, rescaling
from srl.rl.functions.dueling_network import create_dueling_network_layers
from srl.rl.functions.model import ImageLayerType, create_input_layers_lstm_stateful
from srl.rl.memory import factory
from srl.rl.registory import register
from tensorflow.keras import layers as kl

"""
DQN
    window_length               : -
    Target Network              : o
    Huber loss function         : o
    Delay update Target Network : o
    Experience Replay  : o
    Frame skip         : -
    Annealing e-greedy : x
    Reward clip        : x
    Image preprocessor : -
Rainbow
    Double DQN                  : o (config selection)
    Priority Experience Replay  : o (config selection)
    Dueling Network             : o (config selection)
    Multi-Step learning(retrace): o (config selection)
    Noisy Network               : x
    Categorical DQN             : x
Recurrent Replay Distributed DQN(R2D2)
    LSTM                     : o
    Value function rescaling : o
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(DiscreteActionConfig):

    test_epsilon: float = 0
    epsilon: float = 0.1

    # model
    dense_units: int = 512
    lstm_units: int = 512
    image_layer_type: ImageLayerType = ImageLayerType.DQN
    burnin: int = 40

    gamma: float = 0.99  # 割引率
    lr: float = 0.001  # 学習率
    batch_size: int = 32
    target_model_update_interval: int = 100

    # double dqn
    enable_double_dqn: bool = True

    # retrace
    multisteps: int = 1
    retrace_h: float = 1.0

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

    dummy_state_val: float = 0.0

    @staticmethod
    def getName() -> str:
        return "R2D2"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.burnin >= 0
        assert self.multisteps >= 1
        assert self.memory_warmup_size < self.capacity
        assert self.batch_size < self.memory_warmup_size


register(Config, __name__)


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _QNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()
        # input_shape: (batch_size, input_sequence(timestamps), observation_shape)

        # timestamps=1(stateful)
        in_state, c = create_input_layers_lstm_stateful(
            config.batch_size,
            1,
            config.env_observation_shape,
            config.env_observation_type,
            config.image_layer_type,
        )

        # lstm
        c = kl.LSTM(config.lstm_units, stateful=True, name="lstm")(c)

        if config.enable_dueling_network:
            c = create_dueling_network_layers(
                c, config.nb_actions, config.dueling_dense_units, config.dueling_network_type
            )
        else:
            c = kl.Dense(config.dense_units, activation="relu", kernel_initializer="he_normal")(c)
            c = kl.Dense(config.nb_actions, kernel_initializer="truncated_normal")(c)

        self.model = keras.Model(in_state, c)
        self.lstm_layer = self.model.get_layer("lstm")

        # 重みを初期化
        in_shape = (1,) + config.env_observation_shape
        dummy_state = np.zeros(shape=(config.batch_size,) + in_shape, dtype=np.float32)
        val, _ = self(dummy_state, None)
        assert val.shape == (config.batch_size, config.nb_actions)

    def call(self, state, hidden_states):
        self.lstm_layer.reset_states(hidden_states)
        return self.model(state), self.get_hidden_state()

    def get_hidden_state(self):
        return [self.lstm_layer.states[0].numpy(), self.lstm_layer.states[1].numpy()]

    def init_hidden_state(self):
        self.lstm_layer.reset_states()
        return self.get_hidden_state()


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
        self.invalid_memory = deque(maxlen=self.config.capacity)

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

        self.optimizer = keras.optimizers.Adam(learning_rate=self.config.lr)
        self.loss = keras.losses.Huber()

        self.train_count = 0

    def get_train_count(self):
        return self.train_count

    def train(self):

        if self.memory.length() < self.config.memory_warmup_size:
            return {}

        indexes, batchs, weights = self.memory.sample(self.train_count)
        td_errors, loss = self._train_on_batchs(batchs, weights)
        priorities = np.abs(td_errors) + 0.0001
        self.memory.update(indexes, batchs, priorities)

        # invalid action
        mem_invalid_len = self.memory.length_invalid()
        if mem_invalid_len > self.config.memory_warmup_size:
            batchs = self.memory.sample_invalid()
            self._train_on_batchs(batchs, [1 for _ in range(self.config.batch_size)])

        # targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.set_weights(self.parameter.q_online.get_weights())

        self.train_count += 1
        return {
            "loss": loss,
            "mem2": mem_invalid_len,
            "priority": np.mean(priorities),
        }

    def _train_on_batchs(self, batchs, weights):

        # burnin=2
        # multisteps=3
        # states  [0,1,2,3,4,5,6]
        # burnin   o o
        # state        o,o
        # n_state1       o,o
        # n_state2         o,o
        # n_state3           o,o

        # (batch, dict[x], multisteps) -> (multisteps, batch, x)
        states_list = []
        for i in range(self.config.multisteps + 1):
            states_list.append(np.asarray([[b["states"][i]] for b in batchs]))
        actions_list = []
        mu_probs_list = []
        rewards_list = []
        dones_list = []
        for i in range(self.config.multisteps):
            actions_list.append([b["actions"][i] for b in batchs])
            rewards_list.append([b["rewards"][i] for b in batchs])
            mu_probs_list.append([b["probs"][i] for b in batchs])
            dones_list.append([b["dones"][i] for b in batchs])

        # hidden_states
        states_h = []
        states_c = []
        for b in batchs:
            states_h.append(b["hidden_states"][0])
            states_c.append(b["hidden_states"][1])
        hidden_states = [np.asarray(states_h), np.asarray(states_c)]

        # burnin
        for i in range(self.config.burnin):
            burnin_state = np.asarray([[b["burnin_states"][i]] for b in batchs])
            _, hidden_states = self.parameter.q_online(burnin_state, hidden_states)

        _, _, td_error, _, loss = self._train_steps(
            states_list,
            actions_list,
            mu_probs_list,
            rewards_list,
            dones_list,
            weights,
            hidden_states,
            0,
        )
        return td_error, loss

    # Q値(LSTM hidden states)の予測はforward、td_error,retraceはbackで予測する必要あり
    def _train_steps(
        self,
        states_list,
        actions_list,
        mu_probs_list,
        rewards_list,
        dones_list,
        weights,
        hidden_states,
        idx,
    ):

        # 最後
        if idx == self.config.multisteps:
            n_states = states_list[idx]
            n_q, _ = self.parameter.q_online(n_states, hidden_states)
            n_q_target, _ = self.parameter.q_target(n_states, hidden_states)
            n_q = tf.stop_gradient(n_q).numpy()
            n_q_target = tf.stop_gradient(n_q_target).numpy()
            return n_q, n_q_target, np.zeros(self.config.batch_size), 1.0, 0

        states = states_list[idx]
        n_states = states_list[idx + 1]
        actions = actions_list[idx]
        dones = dones_list[idx]
        rewards = rewards_list[idx]
        mu_probs = mu_probs_list[idx]

        q_target, _ = self.parameter.q_target(states, hidden_states)
        q_target = tf.stop_gradient(q_target).numpy()
        with tf.GradientTape() as tape:
            q, n_hidden_states = self.parameter.q_online(states, hidden_states)

            n_q, n_q_target, n_td_error, retrace, _ = self._train_steps(
                states_list,
                actions_list,
                mu_probs_list,
                rewards_list,
                dones_list,
                weights,
                n_hidden_states,
                idx + 1,
            )

            target_q = []
            for i in range(self.config.batch_size):
                if dones[i]:
                    gain = rewards[i]
                else:
                    # DoubleDQN: indexはonlineQから選び、値はtargetQを選ぶ
                    if self.config.enable_double_dqn:
                        n_act_idx = np.argmax(n_q[i])
                    else:
                        n_act_idx = np.argmax(n_q_target[i])
                    maxq = n_q_target[i][n_act_idx]
                    maxq = inverse_rescaling(maxq)
                    gain = rewards[i] + self.config.gamma * maxq
                gain = rescaling(gain)
                target_q.append(gain)
            target_q = np.asarray(target_q)

            # retrace
            _retrace = []
            for i in range(self.config.batch_size):
                pi_probs = calc_epsilon_greedy_probs(
                    n_q[i],
                    [a for a in range(self.config.nb_actions)],
                    0.0,
                    self.config.nb_actions,
                )
                r = self.config.retrace_h * np.minimum(1, pi_probs[actions[i]] / mu_probs[i])
                _retrace.append(r)
            retrace *= np.asarray(_retrace)

            target_q += self.config.gamma * retrace * n_td_error

            # 現在選んだアクションのQ値
            action_onehot = tf.one_hot(actions, self.config.nb_actions)
            q_onehot = tf.reduce_sum(q * action_onehot, axis=1)

            loss = self.loss(target_q * weights, q_onehot * weights)

        grads = tape.gradient(loss, self.parameter.q_online.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.q_online.trainable_variables))

        n_td_error = (target_q - q_onehot).numpy() + self.config.gamma * retrace * n_td_error
        q = tf.stop_gradient(q).numpy()
        return q, q_target, n_td_error, retrace, loss.numpy()


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

    def on_reset(self, state: np.ndarray, valid_actions: List[int], _) -> None:
        self.recent_states = [self.dummy_state for _ in range(self.config.burnin + self.config.multisteps + 1)]
        self.recent_actions = [random.randint(0, self.config.nb_actions - 1) for _ in range(self.config.multisteps)]
        self.recent_probs = [1.0 / self.config.nb_actions for _ in range(self.config.multisteps)]
        self.recent_rewards = [0.0 for _ in range(self.config.multisteps)]
        self.recent_done = [False for _ in range(self.config.multisteps)]
        self.recent_valid_actions = [[] for _ in range(self.config.multisteps + 1)]

        self.hidden_state = self.parameter.q_online.init_hidden_state()
        self.recent_hidden_states = [
            [self.hidden_state[0][0], self.hidden_state[1][0]]
            for _ in range(self.config.burnin + self.config.multisteps + 1)
        ]

        self.recent_states.pop(0)
        self.recent_states.append(state.astype(np.float32))
        self.recent_valid_actions.pop(0)
        self.recent_valid_actions.append(valid_actions)

    def policy(self, state: np.ndarray, valid_actions: List[int], _) -> Tuple[int, Any]:
        state = np.asarray([[state]] * self.config.batch_size)
        q, self.hidden_state = self.parameter.q_online(state, self.hidden_state)
        q = q[0].numpy()

        if self.training:
            epsilon = self.config.epsilon
        else:
            epsilon = self.config.test_epsilon

        probs = calc_epsilon_greedy_probs(q, valid_actions, epsilon, self.config.nb_actions)
        action = random_choice_by_probs(probs)

        return action, (action, probs[action], q[action])

    def on_step(
        self,
        state: np.ndarray,
        action_: Any,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        valid_actions: List[int],
        next_valid_actions: List[int],
        _,
    ):
        if not self.training:
            return {}

        action = action_[0]
        prob = action_[1]
        q = action_[2]

        if self.invalid_action_reward > reward - 1:
            self.invalid_action_reward = reward - 1
            self.memory.clear_invalid()

        self.recent_states.pop(0)
        self.recent_states.append(next_state.astype(np.float32))
        self.recent_actions.pop(0)
        self.recent_actions.append(action)
        self.recent_probs.pop(0)
        self.recent_probs.append(prob)
        self.recent_rewards.pop(0)
        self.recent_rewards.append(reward)
        self.recent_done.pop(0)
        self.recent_done.append(done)
        self.recent_valid_actions.pop(0)
        self.recent_valid_actions.append(next_valid_actions)
        self.recent_hidden_states.pop(0)
        self.recent_hidden_states.append(
            [
                self.hidden_state[0][0],
                self.hidden_state[1][0],
            ]
        )

        priority = self._add_memory(q, None)

        if done:
            # 残りstepも追加
            for _ in range(len(self.recent_rewards) - 1):
                self.recent_states.pop(0)
                self.recent_states.append(self.dummy_state)
                self.recent_actions.pop(0)
                self.recent_actions.append(random.randint(0, self.config.nb_actions - 1))
                self.recent_probs.pop(0)
                self.recent_probs.append(1.0 / self.config.nb_actions)
                self.recent_rewards.pop(0)
                self.recent_rewards.append(0.0)
                self.recent_done.pop(0)
                self.recent_done.append(True)
                self.recent_valid_actions.pop(0)
                self.recent_valid_actions.append([])
                self.recent_hidden_states.pop(0)

                self._add_memory(q, priority)

        # --- valid action
        if False:  # TODO
            states = self.recent_states[:]
            states[-1] = self.dummy_state
            bundle_states = self.recent_bundle_states[:]
            bundle_states[-1] = states
            rewards = self.recent_rewards[:]
            rewards[-1] = self.invalid_action_reward
            probs = self.recent_probs[:]
            probs[-1] = 1.0
            dones = self.recent_done[:]
            dones[-1] = True
            for a in range(self.config.nb_actions):
                if a in valid_actions:
                    continue
                actions = self.recent_actions[:]
                actions[-1] = a

                batch = {
                    "states": bundle_states,
                    "actions": actions,
                    "probs": probs,
                    "rewards": rewards,
                    "dones": dones,
                    "valid_actions": self.recent_valid_actions[:],
                }
                self.memory.add_invalid(batch)

        return {"priority": priority}

    def _add_memory(self, q, priority):

        batch = {
            "states": self.recent_states[self.config.burnin :],
            "actions": self.recent_actions[:],
            "probs": self.recent_probs[:],
            "rewards": self.recent_rewards[:],
            "dones": self.recent_done[:],
            "valid_actions": self.recent_valid_actions[:],
            "burnin_states": self.recent_states[: self.config.burnin],
            "hidden_states": self.recent_hidden_states[0],
        }

        # priority
        if priority is None:
            if self.config.memory_name == "ReplayMemory":
                priority = 1
            else:
                # TODO
                priority = 1
                # target_q = self.parameter.calc_target_q([batch])[0]
                # priority = abs(target_q - q) + 0.0001

        self.memory.add(batch, priority)
        return priority

    def render(self, state: np.ndarray, valid_actions: List[int], action_to_str) -> None:
        state = np.asarray([[state]] * self.config.batch_size)
        q, self.hidden_state = self.parameter.q_online(state, self.hidden_state)
        q = q[0].numpy()

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
            s += f"{action_to_str(a)}: {q[a]:5.3f}"
            print(s)


if __name__ == "__main__":
    pass

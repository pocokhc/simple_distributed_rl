import random
from dataclasses import dataclass, field
from typing import Any, List

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.rl.algorithms.base_dqn import RLConfig, RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.processor import RLProcessor
from srl.base.rl.registration import register
from srl.base.rl.trainer import RLTrainer
from srl.base.spaces.space import SpaceBase
from srl.rl import functions as funcs
from srl.rl.functions import create_epsilon_list, inverse_rescaling, rescaling
from srl.rl.memories.priority_replay_buffer import PriorityReplayBufferConfig, RLPriorityReplayBuffer
from srl.rl.models.config.dueling_network import DuelingNetworkConfig
from srl.rl.models.config.input_image_block import InputImageBlockConfig
from srl.rl.models.config.input_value_block import InputValueBlockConfig
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig
from srl.rl.tf.model import KerasModelAddedSummary
from srl.utils.common import compare_less_version

kl = keras.layers
v216_older = compare_less_version(tf.__version__, "2.16.0")

"""
・Paper
https://openreview.net/forum?id=r1lyTjAqYX

DQN
    window_length          : -
    Fixed Target Q-Network : o
    Error clipping     : o
    Experience Replay  : o
    Frame skip         : -
    Annealing e-greedy : x
    Reward clip        : x
    Image preprocessor : -
Rainbow
    Double DQN                  : o (config selection)
    Priority Experience Replay  : o (config selection)
    Dueling Network             : o (config selection)
    Multi-Step learning         : x
    Noisy Network               : x
    Categorical DQN             : x
Recurrent Replay Distributed DQN(R2D2)
    LSTM                     : o
    Value function rescaling : o (config selection)
Never Give Up(NGU)
    Retrace          : o (config selection)
Other
    invalid_actions : o

"""


@dataclass
class Config(RLConfig):
    test_epsilon: float = 0
    epsilon: float = 0.1
    actor_epsilon: float = 0.4
    actor_alpha: float = 7.0

    #: Batch size
    batch_size: int = 32
    #: <:ref:`PriorityReplayBufferConfig`>
    memory: PriorityReplayBufferConfig = field(default_factory=lambda: PriorityReplayBufferConfig())

    #: <:ref:`InputValueBlockConfig`>
    input_value_block: InputValueBlockConfig = field(default_factory=lambda: InputValueBlockConfig())
    #: <:ref:`InputImageBlockConfig`>
    input_image_block: InputImageBlockConfig = field(default_factory=lambda: InputImageBlockConfig())
    lstm_units: int = 512
    hidden_block: DuelingNetworkConfig = field(init=False, default_factory=lambda: DuelingNetworkConfig().set_dueling_network())

    # lstm
    burnin: int = 5
    sequence_length: int = 5

    discount: float = 0.997
    lr: float = 0.001
    #: <:ref:`LRSchedulerConfig`>
    lr_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())
    target_model_update_interval: int = 1000

    # double dqn
    enable_double_dqn: bool = True

    # rescale
    enable_rescale: bool = False

    # retrace
    enable_retrace: bool = True
    retrace_h: float = 1.0

    def setup_from_actor(self, actor_num: int, actor_id: int) -> None:
        self.epsilon = create_epsilon_list(
            actor_num,
            epsilon=self.actor_epsilon,
            alpha=self.actor_alpha,
        )[actor_id]

    def set_atari_config(self):
        # model
        self.lstm_units = 512
        self.input_value_block.set(())
        self.input_image_block.set_dqn_block()
        self.hidden_block.set_dueling_network((512,))

        # lstm
        self.burnin = 40
        self.sequence_length = 80

        self.discount = 0.997
        self.lr = 0.0001
        self.batch_size = 64
        self.target_model_update_interval = 2500

        self.enable_double_dqn = True
        self.enable_rescale = True
        self.enable_retrace = False

        self.memory.capacity = 1_000_000
        self.memory.set_proportional(
            alpha=0.9,
            beta_initial=0.6,
            beta_steps=1_000_000,
        )

    def get_name(self) -> str:
        return "R2D2"

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        if prev_observation_space.is_image():
            return self.input_image_block.get_processors()
        return []

    def get_framework(self) -> str:
        return "tensorflow"

    def validate_params(self) -> None:
        super().validate_params()
        if not (self.burnin >= 0):
            raise ValueError(f"assert {self.burnin} >= 0")
        if not (self.sequence_length >= 1):
            raise ValueError(f"assert {self.sequence_length} >= 1")


register(
    Config(),
    __name__ + ":Memory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


class Memory(RLPriorityReplayBuffer):
    pass


class QNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config):
        super().__init__()

        if config.observation_space.is_image():
            self.in_block = config.input_image_block.create_tf_block(config.observation_space, rnn=True)
        else:
            self.in_block = config.input_value_block.create_tf_block(config.observation_space, rnn=True)

        # --- lstm
        self.lstm_layer = kl.LSTM(
            config.lstm_units,
            return_sequences=True,
            return_state=True,
        )

        # out
        self.hidden_block = config.hidden_block.create_tf_block(config.action_space.n, rnn=True)

        # build
        self(self.in_block.create_dummy_data(config.get_dtype("np"), timesteps=config.sequence_length))

    @tf.function()
    def call(self, x, hidden_states=None, training=False):
        return self._call(x, hidden_states, training=training)

    def _call(self, x, hidden_state=None, training=False):
        x = self.in_block(x, training=training)

        # lstm
        x, h, c = self.lstm_layer(x, initial_state=hidden_state, training=training)

        x = self.hidden_block(x, training=training)
        return x, [h, c]

    def get_initial_state(self, batch_size=1):
        if v216_older:
            return self.lstm_layer.cell.get_initial_state(batch_size=batch_size, dtype=self.dtype)
        else:
            return self.lstm_layer.cell.get_initial_state(batch_size)


class Parameter(RLParameter[Config]):
    def setup(self):
        self.q_online = QNetwork(self.config)
        self.q_target = QNetwork(self.config)
        self.q_target.set_weights(self.q_online.get_weights())

    def call_restore(self, data: Any, **kwargs) -> None:
        self.q_online.set_weights(data)
        self.q_target.set_weights(data)

    def call_backup(self, **kwargs) -> Any:
        return self.q_online.get_weights()

    def summary(self, **kwargs):
        self.q_online.summary(**kwargs)


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        lr = self.config.lr_scheduler.apply_tf_scheduler(self.config.lr)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.loss = keras.losses.Huber()

        self.sync_count = 0

    def train(self) -> None:
        batches = self.memory.sample()
        if batches is None:
            return
        batches, weights, update_args = batches

        td_errors, loss = self._train_on_batches(batches, np.array(weights).reshape(-1, 1))
        self.memory.update(update_args, np.array(td_errors), self.train_count)

        # targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.set_weights(self.parameter.q_online.get_weights())
            self.sync_count += 1

        self.train_count += 1
        self.info["loss"] = loss
        self.info["sync"] = self.sync_count

    def _train_on_batches(self, batches, weights):
        # (batch, dict[x], step) -> (batch, step, x)
        burnin_states = []
        step_states = []
        step_actions = []
        for b in batches:
            burnin_states.append([s for s in b["states"][: self.config.burnin]])
            step_states.append([s for s in b["states"][self.config.burnin :]])
            step_actions.append([s for s in b["actions"]])
        burnin_states = np.asarray(burnin_states)
        step_states = np.asarray(step_states)

        # (batch, step, x)
        step_actions_onehot = tf.one_hot(step_actions, self.config.action_space.n, axis=2)

        # (batch, list, hidden) -> (list, batch, hidden)
        states_h = []
        states_c = []
        for b in batches:
            states_h.append(b["hidden_states"][0])
            states_c.append(b["hidden_states"][1])
        hidden_states = [tf.stack(states_h), tf.stack(states_c)]
        hidden_states_t = hidden_states

        # burn-in
        if self.config.burnin > 0:
            _, hidden_states = self.parameter.q_online(burnin_states, hidden_states)  # type:ignore , ignore check "None"
            _, hidden_states_t = self.parameter.q_target(burnin_states, hidden_states_t)  # type:ignore , ignore check "None"

        # targetQ
        q_target, _ = self.parameter.q_target(step_states, hidden_states_t)  # type:ignore , ignore check "None"
        q_target = q_target.numpy()

        # --- 勾配 + targetQを計算
        td_errors_list = []
        with tf.GradientTape() as tape:
            q, _ = self.parameter.q_online(step_states, hidden_states, training=True)  # type:ignore , ignore check "None"
            frozen_q = tf.stop_gradient(q).numpy()  # type:ignore , ignore check "None"

            # 最後は学習しないので除く
            tf.stop_gradient(q[:, -1, ...])
            q = q[:, :-1, ...]

            # --- TargetQを計算
            target_q_list = []
            for idx, b in enumerate(batches):
                retrace = 1
                next_td_error = 0
                td_errors = []
                target_q = []

                # 後ろから計算
                for t in reversed(range(self.config.sequence_length)):
                    action = b["actions"][t]
                    mu_prob = b["probs"][t]
                    reward = b["rewards"][t]
                    done = b["dones"][t]
                    invalid_actions = b["invalid_actions"][t]
                    next_invalid_actions = b["invalid_actions"][t + 1]

                    if done:
                        gain = reward
                    else:
                        # DoubleDQN: indexはonlineQから選び、値はtargetQを選ぶ
                        if self.config.enable_double_dqn:
                            n_q = frozen_q[idx][t + 1]
                        else:
                            n_q = q_target[idx][t + 1]
                        n_q = [(-np.inf if a in next_invalid_actions else v) for a, v in enumerate(n_q)]
                        n_act_idx = np.argmax(n_q)
                        maxq = q_target[idx][t + 1][n_act_idx]
                        if self.config.enable_rescale:
                            maxq = inverse_rescaling(maxq)
                        gain = reward + self.config.discount * maxq
                    if self.config.enable_rescale:
                        gain = rescaling(gain)
                    target_q.insert(0, gain + retrace * next_td_error)

                    td_error = gain - frozen_q[idx][t][action]
                    td_errors.append(td_error)
                    if self.config.enable_retrace:
                        # TDerror
                        next_td_error = td_error

                        # retrace
                        pi_probs = funcs.calc_epsilon_greedy_probs(
                            frozen_q[idx][t],
                            invalid_actions,
                            0.0,
                            self.config.action_space.n,
                        )
                        pi_prob = pi_probs[action]
                        _r = self.config.retrace_h * np.minimum(1, pi_prob / mu_prob)
                        retrace *= self.config.discount * _r
                target_q_list.append(target_q)
                td_errors_list.append(np.mean(td_errors))
            target_q_list = np.asarray(target_q_list)

            # --- update Q
            q_onehot = tf.reduce_sum(q * step_actions_onehot, axis=2)
            loss = self.loss(target_q_list * weights, q_onehot * weights)
            loss += tf.reduce_sum(self.parameter.q_online.losses)

        grads = tape.gradient(loss, self.parameter.q_online.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.q_online.trainable_variables))

        return td_errors_list, loss.numpy()


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker[Config, Parameter, Memory]):
    def __init__(self, *args):
        super().__init__(*args)

        self.dummy_state = np.zeros(self.config.observation_space.shape, dtype=np.float32)

    def on_reset(self, worker):
        # states : burnin + sequence_length + next_state
        # actions: sequence_length
        # probs  : sequence_length
        # rewards: sequence_length
        # done   : sequence_length
        # invalid_actions: sequence_length + next_invalid_actions
        # hidden_state   : burnin + sequence_length + next_state

        self._recent_states = [self.dummy_state for _ in range(self.config.burnin + self.config.sequence_length + 1)]
        self.recent_actions = [random.randint(0, self.config.action_space.n - 1) for _ in range(self.config.sequence_length)]
        self.recent_probs = [1.0 / self.config.action_space.n for _ in range(self.config.sequence_length)]
        self.recent_rewards = [0.0 for _ in range(self.config.sequence_length)]
        self.recent_done = [False for _ in range(self.config.sequence_length)]
        self.recent_invalid_actions = [[] for _ in range(self.config.sequence_length + 1)]

        self.hidden_state = self.parameter.q_online.get_initial_state()
        self.recent_hidden_states = [[self.hidden_state[0][0], self.hidden_state[1][0]] for _ in range(self.config.burnin + self.config.sequence_length + 1)]

        self._recent_states.pop(0)
        self._recent_states.append(worker.state.astype(np.float32))
        self.recent_invalid_actions.pop(0)
        self.recent_invalid_actions.append(worker.invalid_actions)

        # TD誤差を計算するか
        if not self.distributed:
            self._calc_td_error = False
        elif not self.config.memory.requires_priority():
            self._calc_td_error = False
        else:
            self._calc_td_error = True
            self._history_batch = []

    def policy(self, worker) -> int:
        state = worker.state[np.newaxis, np.newaxis, ...]  # (batch, time step, ...)
        q, self.hidden_state = self.parameter.q_online(state, self.hidden_state)
        q = q[0][0].numpy()  # (batch, time step, action_num)

        if self.training:
            epsilon = self.config.epsilon
        else:
            epsilon = self.config.test_epsilon

        probs = funcs.calc_epsilon_greedy_probs(q, worker.invalid_actions, epsilon, self.config.action_space.n)
        self.action = funcs.random_choice_by_probs(probs)

        self.prob = probs[self.action]
        self.q = q
        return self.action

    def on_step(self, worker):
        if not self.training:
            return

        self._recent_states.pop(0)
        self._recent_states.append(worker.next_state.astype(np.float32))
        self.recent_actions.pop(0)
        self.recent_actions.append(self.action)
        self.recent_probs.pop(0)
        self.recent_probs.append(self.prob)
        self.recent_rewards.pop(0)
        self.recent_rewards.append(worker.reward)
        self.recent_done.pop(0)
        self.recent_done.append(worker.terminated)
        self.recent_invalid_actions.pop(0)
        self.recent_invalid_actions.append(worker.next_invalid_actions)
        self.recent_hidden_states.pop(0)
        self.recent_hidden_states.append(
            [
                self.hidden_state[0][0],
                self.hidden_state[1][0],
            ]
        )

        if self._calc_td_error:
            calc_info = {
                "q": self.q[self.action],
                "reward": worker.reward,
            }
        else:
            calc_info = None

        self._add_memory(calc_info)

        if worker.done:
            # 残りstepも追加
            for _ in range(len(self.recent_rewards) - 1):
                self._recent_states.pop(0)
                self._recent_states.append(self.dummy_state)
                self.recent_actions.pop(0)
                self.recent_actions.append(random.randint(0, self.config.action_space.n - 1))
                self.recent_probs.pop(0)
                self.recent_probs.append(1.0 / self.config.action_space.n)
                self.recent_rewards.pop(0)
                self.recent_rewards.append(0.0)
                self.recent_done.pop(0)
                self.recent_done.append(True)
                self.recent_invalid_actions.pop(0)
                self.recent_invalid_actions.append([])
                self.recent_hidden_states.pop(0)

                self._add_memory(
                    {
                        "q": self.q[self.action],
                        "reward": 0.0,
                    }
                )

                if self._calc_td_error:
                    # TD誤差を計算してメモリに送る
                    # targetQはモンテカルロ法
                    reward = 0
                    for batch, info in reversed(self._history_batch):
                        if self.config.enable_rescale:
                            _r = inverse_rescaling(reward)
                        else:
                            _r = reward
                        reward = info["reward"] + self.config.discount * _r
                        if self.config.enable_rescale:
                            reward = rescaling(reward)

                        priority = abs(reward - info["q"])
                        self.memory.add(batch, priority)

    def _add_memory(self, calc_info):
        batch = {
            "states": self._recent_states[:],
            "actions": self.recent_actions[:],
            "probs": self.recent_probs[:],
            "rewards": self.recent_rewards[:],
            "dones": self.recent_done[:],
            "invalid_actions": self.recent_invalid_actions[:],
            "hidden_states": self.recent_hidden_states[0],
        }

        if self._calc_td_error:
            # エピソード最後に計算してメモリに送る
            self._history_batch.append([batch, calc_info])
        else:
            # 計算する必要がない場合はそのままメモリに送る
            self.memory.add(batch, None)

    def render_terminal(self, worker, **kwargs) -> None:
        # パラメータを予測するとhidden_stateが変わってしまうの予測はしない
        q = self.q
        maxa = np.argmax(q)
        if self.config.enable_rescale:
            q = inverse_rescaling(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        funcs.render_discrete_action(int(maxa), self.config.action_space, worker.env, _render_sub)

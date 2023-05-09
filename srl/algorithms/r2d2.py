import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as kl

from srl.base.define import EnvObservationType, RLObservationType
from srl.base.rl.algorithms.discrete_action import DiscreteActionConfig, DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.memory import IPriorityMemoryConfig
from srl.base.rl.model import IImageBlockConfig
from srl.base.rl.processor import Processor
from srl.base.rl.processors.image_processor import ImageProcessor
from srl.base.rl.registration import register
from srl.base.rl.remote_memory import PriorityExperienceReplay
from srl.rl.functions.common import (
    calc_epsilon_greedy_probs,
    create_epsilon_list,
    inverse_rescaling,
    random_choice_by_probs,
    render_discrete_action,
    rescaling,
)
from srl.rl.memories.config import ProportionalMemoryConfig
from srl.rl.models.dqn.dqn_image_block_config import DQNImageBlockConfig
from srl.rl.models.tf.dueling_network import DuelingNetworkBlock
from srl.rl.models.tf.input_block import InputBlock

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


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(DiscreteActionConfig):
    test_epsilon: float = 0
    epsilon: float = 0.1
    actor_epsilon: float = 0.4
    actor_alpha: float = 7.0

    # model
    image_block_config: IImageBlockConfig = field(default_factory=lambda: DQNImageBlockConfig())
    lstm_units: int = 512
    hidden_layer_sizes: Tuple[int, ...] = (512,)
    activation: str = "relu"

    # lstm
    burnin: int = 5
    sequence_length: int = 5

    discount: float = 0.997
    lr: float = 0.001
    batch_size: int = 32
    memory_warmup_size: int = 1000
    target_model_update_interval: int = 1000

    # double dqn
    enable_double_dqn: bool = True

    # rescale
    enable_rescale: bool = False

    # retrace
    enable_retrace: bool = True
    retrace_h: float = 1.0

    # DuelingNetwork
    enable_dueling_network: bool = True
    dueling_network_type: str = "average"

    # memory
    memory: IPriorityMemoryConfig = field(default_factory=lambda: ProportionalMemoryConfig())

    # other
    dummy_state_val: float = 0.0

    def set_config_by_actor(self, actor_num: int, actor_id: int) -> None:
        self.epsilon = create_epsilon_list(actor_num, epsilon=self.actor_epsilon, alpha=self.actor_alpha)[actor_id]

    # 論文のハイパーパラメーター
    def set_atari_config(self):
        # model
        self.lstm_units = 512
        self.hidden_layer_sizes = (512,)

        # lstm
        self.burnin = 40
        self.sequence_length = 80

        self.discount = 0.997
        self.lr = 0.0001
        self.batch_size = 64
        self.target_model_update_interval = 2500

        self.enable_double_dqn = True
        self.enable_dueling_network = True
        self.enable_rescale = True
        self.enable_retrace = False

        self.memory = ProportionalMemoryConfig(
            capacity=1_000_000,
            alpha=0.9,
            beta_initial=0.6,
            beta_steps=1_000_000,
        )

    def set_processor(self) -> List[Processor]:
        return [
            ImageProcessor(
                image_type=EnvObservationType.GRAY_2ch,
                resize=(84, 84),
                enable_norm=True,
            )
        ]

    @property
    def observation_type(self) -> RLObservationType:
        return RLObservationType.CONTINUOUS

    def getName(self) -> str:
        return "R2D2"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.burnin >= 0
        assert self.sequence_length >= 1
        assert self.memory_warmup_size < self.memory.get_capacity()
        assert self.batch_size < self.memory_warmup_size
        assert len(self.hidden_layer_sizes) > 0


register(
    Config(),
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
        super().init(self.config.memory)


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _QNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        # --- in block
        self.in_block = InputBlock(
            config.observation_shape,
            config.env_observation_type,
            enable_time_distributed_layer=True,
        )
        self.use_image_layer = self.in_block.use_image_layer

        # image
        if self.use_image_layer:
            self.image_block = config.image_block_config.create_block_tf(enable_time_distributed_layer=True)
            self.image_flatten = kl.TimeDistributed(kl.Flatten())

        # --- lstm
        self.lstm_layer = kl.LSTM(config.lstm_units, return_sequences=True, return_state=True)

        # hidden
        self.hidden_layers = []
        for i in range(len(config.hidden_layer_sizes) - 1):
            self.hidden_layers.append(
                kl.TimeDistributed(
                    kl.Dense(
                        config.hidden_layer_sizes[i],
                        activation=config.activation,
                        kernel_initializer="he_normal",
                    )
                )
            )

        # out
        self.enable_dueling_network = config.enable_dueling_network
        if config.enable_dueling_network:
            self.dueling_block = DuelingNetworkBlock(
                config.action_num,
                config.hidden_layer_sizes[-1],
                config.dueling_network_type,
                activation=config.activation,
                enable_time_distributed_layer=True,
            )
        else:
            self.out_layers = [
                kl.TimeDistributed(
                    kl.Dense(
                        config.hidden_layer_sizes[-1], activation=config.activation, kernel_initializer="he_normal"
                    )
                ),
                kl.TimeDistributed(
                    kl.Dense(
                        config.action_num, kernel_initializer="truncated_normal", bias_initializer="truncated_normal"
                    )
                ),
            ]

        # build
        self.build((None, config.sequence_length) + config.observation_shape)

    def call(self, state, hidden_state=None, training=False):
        x = self.in_block(state, training=training)
        if self.use_image_layer:
            x = self.image_block(x, training=training)
            x = self.image_flatten(x)

        # lstm
        x, h, c = self.lstm_layer(x, initial_state=hidden_state, training=training)

        # hidden
        for layer in self.hidden_layers:
            x = layer(x, training=training)

        # out
        if self.enable_dueling_network:
            x = self.dueling_block(x, training=training)
        else:
            for layer in self.out_layers:
                x = layer(x, training=training)

        return x, [h, c]

    def init_hidden_state(self):
        return self.lstm_layer.cell.get_initial_state(batch_size=1, dtype=tf.float32)

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def summary(self, name="", **kwargs):
        if hasattr(self.in_block, "init_model_graph"):
            self.in_block.init_model_graph()
        if self.in_block.use_image_layer and hasattr(self.image_block, "init_model_graph"):
            self.image_block.init_model_graph()
        if self.enable_dueling_network and hasattr(self.dueling_block, "init_model_graph"):
            self.dueling_block.init_model_graph()

        x = kl.Input(self.__input_shape[1:])
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self.call(x), name=name)
        model.summary(**kwargs)


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)

        self.q_online = _QNetwork(self.config)
        self.q_target = _QNetwork(self.config)

    def call_restore(self, data: Any, **kwargs) -> None:
        self.q_online.set_weights(data)
        self.q_target.set_weights(data)

    def call_backup(self, **kwargs) -> Any:
        return self.q_online.get_weights()

    def summary(self, **kwargs):
        self.q_online.summary(**kwargs)


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

        indices, batchs, weights = self.remote_memory.sample(self.config.batch_size, self.train_count)
        td_errors, loss = self._train_on_batchs(batchs, np.array(weights).reshape(-1, 1))
        self.remote_memory.update(indices, batchs, td_errors)

        # targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.set_weights(self.parameter.q_online.get_weights())
            self.sync_count += 1

        self.train_count += 1
        return {"loss": loss, "sync": self.sync_count}

    def _train_on_batchs(self, batchs, weights):
        # (batch, dict[x], step) -> (batch, step, x)
        burnin_states = []
        step_states = []
        step_actions = []
        for b in batchs:
            burnin_states.append([s for s in b["states"][: self.config.burnin]])
            step_states.append([s for s in b["states"][self.config.burnin :]])
            step_actions.append([s for s in b["actions"]])
        burnin_states = np.asarray(burnin_states)
        step_states = np.asarray(step_states)

        # (batch, step, x)
        step_actions_onehot = tf.one_hot(step_actions, self.config.action_num, axis=2)

        # (batch, list, hidden) -> (list, batch, hidden)
        states_h = []
        states_c = []
        for b in batchs:
            states_h.append(b["hidden_states"][0])
            states_c.append(b["hidden_states"][1])
        hidden_states = [tf.stack(states_h), tf.stack(states_c)]
        hidden_states_t = hidden_states

        # burn-in
        if self.config.burnin > 0:
            _, hidden_states = self.parameter.q_online(burnin_states, hidden_states)
            _, hidden_states_t = self.parameter.q_target(burnin_states, hidden_states_t)

        # targetQ
        q_target, _ = self.parameter.q_target(step_states, hidden_states_t)
        q_target = q_target.numpy()

        # --- 勾配 + targetQを計算
        td_errors_list = []
        with tf.GradientTape() as tape:
            q, _ = self.parameter.q_online(step_states, hidden_states, training=True)
            frozen_q = tf.stop_gradient(q).numpy()

            # 最後は学習しないので除く
            tf.stop_gradient(q[:, -1, ...])
            q = q[:, :-1, ...]

            # --- TargetQを計算
            target_q_list = []
            for idx, b in enumerate(batchs):
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
                        pi_probs = calc_epsilon_greedy_probs(
                            frozen_q[idx][t],
                            invalid_actions,
                            0.0,
                            self.config.action_num,
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
class Worker(DiscreteActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config = cast(Config, self.config)
        self.parameter = cast(Parameter, self.parameter)
        self.remote_memory = cast(RemoteMemory, self.remote_memory)

        self.dummy_state = np.full(self.config.observation_shape, self.config.dummy_state_val, dtype=np.float32)

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> dict:
        # states : burnin + sequence_length + next_state
        # actions: sequence_length
        # probs  : sequence_length
        # rewards: sequence_length
        # done   : sequence_length
        # invalid_actions: sequence_length + next_invalid_actions
        # hidden_state   : burnin + sequence_length + next_state

        self._recent_states = [self.dummy_state for _ in range(self.config.burnin + self.config.sequence_length + 1)]
        self.recent_actions = [
            random.randint(0, self.config.action_num - 1) for _ in range(self.config.sequence_length)
        ]
        self.recent_probs = [1.0 / self.config.action_num for _ in range(self.config.sequence_length)]
        self.recent_rewards = [0.0 for _ in range(self.config.sequence_length)]
        self.recent_done = [False for _ in range(self.config.sequence_length)]
        self.recent_invalid_actions = [[] for _ in range(self.config.sequence_length + 1)]

        self.hidden_state = self.parameter.q_online.init_hidden_state()
        self.recent_hidden_states = [
            [self.hidden_state[0][0], self.hidden_state[1][0]]
            for _ in range(self.config.burnin + self.config.sequence_length + 1)
        ]

        self._recent_states.pop(0)
        self._recent_states.append(state.astype(np.float32))
        self.recent_invalid_actions.pop(0)
        self.recent_invalid_actions.append(invalid_actions)

        # TD誤差を計算するか
        if self.config.memory.is_replay_memory():
            self._calc_td_error = False
        elif not self.distributed:
            self._calc_td_error = False
        else:
            self._calc_td_error = True
            self._history_batch = []

        return {}

    def call_policy(self, state: np.ndarray, invalid_actions: List[int]) -> Tuple[int, dict]:
        state = state[np.newaxis, np.newaxis, ...]  # (batch, time step, ...)
        q, self.hidden_state = self.parameter.q_online(state, self.hidden_state)
        q = q[0][0].numpy()  # (batch, time step, action_num)

        if self.training:
            epsilon = self.config.epsilon
        else:
            epsilon = self.config.test_epsilon

        probs = calc_epsilon_greedy_probs(q, invalid_actions, epsilon, self.config.action_num)
        self.action = random_choice_by_probs(probs)

        self.prob = probs[self.action]
        self.q = q
        return self.action, {}

    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        next_invalid_actions: List[int],
    ) -> Dict:
        if not self.training:
            return {}

        self._recent_states.pop(0)
        self._recent_states.append(next_state.astype(np.float32))
        self.recent_actions.pop(0)
        self.recent_actions.append(self.action)
        self.recent_probs.pop(0)
        self.recent_probs.append(self.prob)
        self.recent_rewards.pop(0)
        self.recent_rewards.append(reward)
        self.recent_done.pop(0)
        self.recent_done.append(done)
        self.recent_invalid_actions.pop(0)
        self.recent_invalid_actions.append(next_invalid_actions)
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
                "reward": reward,
            }
        else:
            calc_info = None

        self._add_memory(calc_info)

        if done:
            # 残りstepも追加
            for _ in range(len(self.recent_rewards) - 1):
                self._recent_states.pop(0)
                self._recent_states.append(self.dummy_state)
                self.recent_actions.pop(0)
                self.recent_actions.append(random.randint(0, self.config.action_num - 1))
                self.recent_probs.pop(0)
                self.recent_probs.append(1.0 / self.config.action_num)
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

                        td_error = reward - info["q"]
                        self.remote_memory.add(batch, td_error)

        self.remote_memory.on_step(reward, done)
        return {}

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
            self.remote_memory.add(batch, None)

    def render_terminal(self, env, worker, **kwargs) -> None:
        invalid_actions = self.recent_invalid_actions[-1]
        # パラメータを予測するとhidden_stateが変わってしまうの予測はしない
        q = self.q
        maxa = np.argmax(q)
        if self.config.enable_rescale:
            q = inverse_rescaling(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        render_discrete_action(invalid_actions, maxa, env, _render_sub)

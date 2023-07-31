import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import EnvObservationTypes, RLTypes
from srl.base.rl.algorithms.discrete_action import DiscreteActionWorker
from srl.base.rl.base import RLParameter, RLTrainer
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import Processor
from srl.base.rl.registration import register
from srl.rl.functions.common import (
    calc_epsilon_greedy_probs,
    create_epsilon_list,
    inverse_rescaling,
    random_choice_by_probs,
    render_discrete_action,
    rescaling,
)
from srl.rl.memories.priority_experience_replay import PriorityExperienceReplay, PriorityExperienceReplayConfig
from srl.rl.models.dqn.tf.dqn_image_block import DQNImageBlock
from srl.rl.models.tf.dueling_network import DuelingNetworkBlock
from srl.rl.processors.image_processor import ImageProcessor

kl = keras.layers

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


def create_input_layer_stateful_lstm(
    batch_size: int,
    observation_shape: Tuple[int, ...],
    observation_type: EnvObservationTypes,
) -> Tuple[kl.Layer, kl.Layer, bool]:
    """状態の入力レイヤーを作成して返します。
    input_sequence は1で固定します。

    Args:
        batch_size (int): batch_size
        observation_shape (Tuple[int, ...]): 状態の入力shape
        observation_type (EnvObservationType): 状態が何かを表すEnvObservationType

    Returns:
        [
            in_layer  (kl.Layer): modelの入力に使うlayerを返します
            out_layer (kl.Layer): modelの続きに使うlayerを返します
            use_image_head (bool):
                Falseの時 out_layer は flatten、
                Trueの時 out_layer は CNN の形式で返ります。
        ]
    """

    # --- input
    input_shape = (1,) + observation_shape
    in_layer = c = kl.Input(batch_input_shape=(batch_size,) + input_shape)

    # --- value head
    if (
        observation_type == EnvObservationTypes.DISCRETE
        or observation_type == EnvObservationTypes.CONTINUOUS
        or observation_type == EnvObservationTypes.UNKNOWN
    ):
        c = kl.TimeDistributed(kl.Flatten())(c)
        return cast(kl.Layer, in_layer), cast(kl.Layer, c), False

    # --- image head
    if observation_type == EnvObservationTypes.GRAY_2ch:
        assert len(input_shape) == 3

        # (timesteps, w, h) -> (timesteps, w, h, 1)
        c = kl.Reshape(input_shape + (-1,))(c)

    elif observation_type == EnvObservationTypes.GRAY_3ch:
        assert len(input_shape) == 4
        assert input_shape[-1] == 1

        # (timesteps, width, height, 1)

    elif observation_type == EnvObservationTypes.COLOR:
        assert len(input_shape) == 4

        # (timesteps, width, height, ch)

    elif observation_type == EnvObservationTypes.SHAPE2:
        assert len(input_shape) == 3

        # (timesteps, width, height) -> (timesteps, width, height, 1)
        c = kl.Reshape(input_shape + (-1,))(c)

    elif observation_type == EnvObservationTypes.SHAPE3:
        assert len(input_shape) == 4

        # (timesteps, n, width, height) -> (timesteps, width, height, n)
        c = kl.Permute((1, 3, 4, 2))(c)

    else:
        raise ValueError(f"unknown observation_type: {observation_type}")

    return cast(kl.Layer, in_layer), cast(kl.Layer, c), True


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(RLConfig, PriorityExperienceReplayConfig):
    test_epsilon: float = 0
    epsilon: float = 0.1
    actor_epsilon: float = 0.4
    actor_alpha: float = 7.0

    # model
    cnn_block: kl.Layer = DQNImageBlock
    cnn_block_kwargs: dict = field(default_factory=lambda: {})
    lstm_units: int = 512
    hidden_layer_sizes: Tuple[int, ...] = (512,)
    activation: str = "relu"

    # lstm
    burnin: int = 5
    sequence_length: int = 5

    discount: float = 0.997
    lr: float = 0.001
    batch_size: int = 32
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

        self.memory.capacity = 1_000_000
        self.memory_warmup_size: int = 80_000
        self.memory.set_proportional_memory(
            alpha=0.9,
            beta_initial=0.6,
            beta_steps=1_000_000,
        )

    def __post_init__(self):
        if self.cnn_block_kwargs is None:
            self.cnn_block_kwargs = {}

    def set_processor(self) -> List[Processor]:
        return [
            ImageProcessor(
                image_type=EnvObservationTypes.GRAY_2ch,
                resize=(84, 84),
                enable_norm=True,
            )
        ]

    @property
    def base_action_type(self) -> RLTypes:
        return RLTypes.DISCRETE

    @property
    def base_observation_type(self) -> RLTypes:
        return RLTypes.CONTINUOUS

    def get_use_framework(self) -> str:
        return "tensorflow"

    def getName(self) -> str:
        return "R2D2_stateful"

    def assert_params(self) -> None:
        super().assert_params()
        assert self.burnin >= 0
        assert self.sequence_length >= 1
        assert self.memory_warmup_size < self.memory.capacity
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
    pass


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _QNetwork(keras.Model):
    def __init__(self, config: Config):
        super().__init__()

        # input_shape: (batch_size, input_sequence(timestamps), observation_shape)
        # timestamps=1(stateful)
        in_state, c, use_image_head = create_input_layer_stateful_lstm(
            config.batch_size,
            config.observation_shape,
            config.env_observation_type,
        )
        if use_image_head:
            c = kl.TimeDistributed(config.cnn_block(**config.cnn_block_kwargs))(c)
            c = kl.TimeDistributed(kl.Flatten())(c)

        # lstm
        c = kl.LSTM(config.lstm_units, stateful=True, name="lstm")(c)

        # hidden layers
        for i in range(len(config.hidden_layer_sizes) - 1):
            c = kl.Dense(
                config.hidden_layer_sizes[i],
                activation=config.activation,
                kernel_initializer="he_normal",
            )(c)

        if config.enable_dueling_network:
            c = DuelingNetworkBlock(
                config.action_num,
                config.hidden_layer_sizes[-1],
                config.dueling_network_type,
                activation=config.activation,
            )(c)
        else:
            c = kl.Dense(config.hidden_layer_sizes[-1], activation=config.activation, kernel_initializer="he_normal")(
                c
            )
            c = kl.Dense(
                config.action_num, kernel_initializer="truncated_normal", bias_initializer="truncated_normal"
            )(c)

        self.model = keras.Model(in_state, c, name="QNetwork")
        self.lstm_layer: kl.LSTM = self.model.get_layer("lstm")

        # 重みを初期化
        in_shape = (1,) + config.observation_shape
        dummy_state = np.zeros(shape=(config.batch_size,) + in_shape, dtype=np.float32)
        val, _ = self(dummy_state, None)  # type: ignore
        assert val.shape == (config.batch_size, config.action_num)

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
        self.config: Config = self.config

        self.q_online = _QNetwork(self.config)
        self.q_target = _QNetwork(self.config)

    def call_restore(self, data: Any, **kwargs) -> None:
        self.q_online.set_weights(data)
        self.q_target.set_weights(data)

    def call_backup(self, **kwargs) -> Any:
        return self.q_online.get_weights()

    def summary(self, **kwargs):
        self.q_online.model.summary(**kwargs)


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------


def _batch_dict_step_to_step_batch(batchs, key, in_step, is_list, to_np):
    _list = []
    for i in range(len(batchs[0][key])):
        if in_step and is_list:
            _list.append([[b[key][i]] for b in batchs])
        elif in_step and not is_list:
            _list.append([[[b[key][i]]] for b in batchs])
        elif not in_step and is_list:
            _list.append([[b[key][i]] for b in batchs])
        elif not in_step and not is_list:
            _list.append([b[key][i] for b in batchs])
    if to_np:
        _list = np.asarray(_list)
    return _list


class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter
        self.remote_memory: RemoteMemory = self.remote_memory

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
        td_errors, loss = self._train_on_batchs(batchs, weights)
        self.remote_memory.update(indices, batchs, np.array(td_errors))

        # targetと同期
        if self.train_count % self.config.target_model_update_interval == 0:
            self.parameter.q_target.set_weights(self.parameter.q_online.get_weights())
            self.sync_count += 1

        self.train_count += 1
        return {"loss": loss, "sync": self.sync_count}

    def _train_on_batchs(self, batchs, weights):
        # (batch, dict[x], step) -> (step, batch, 1, x)
        states_list = _batch_dict_step_to_step_batch(batchs, "states", in_step=True, is_list=True, to_np=True)
        burnin_states = states_list[: self.config.burnin]
        step_states_list = states_list[self.config.burnin :]

        # (batch, dict[x], step) -> (step, batch)
        step_actions_list = _batch_dict_step_to_step_batch(
            batchs, "actions", in_step=False, is_list=False, to_np=False
        )
        step_mu_probs_list = _batch_dict_step_to_step_batch(batchs, "probs", in_step=False, is_list=False, to_np=False)
        step_rewards_list = _batch_dict_step_to_step_batch(
            batchs, "rewards", in_step=False, is_list=False, to_np=False
        )
        step_dones_list = _batch_dict_step_to_step_batch(batchs, "dones", in_step=False, is_list=False, to_np=False)
        step_next_invalid_actions_list = _batch_dict_step_to_step_batch(
            batchs, "invalid_actions", in_step=False, is_list=False, to_np=False
        )

        # (step, batch, x)
        step_actions_onehot_list = tf.one_hot(step_actions_list, self.config.action_num)

        # hidden_states
        states_h = []
        states_c = []
        for b in batchs:
            states_h.append(b["hidden_states"][0])
            states_c.append(b["hidden_states"][1])
        hidden_states = [np.asarray(states_h), np.asarray(states_c)]
        hidden_states_t = [np.asarray(states_h), np.asarray(states_c)]

        # burn-in
        for i in range(self.config.burnin):
            burnin_state = burnin_states[i]
            _, hidden_states = self.parameter.q_online(burnin_state, hidden_states)  # type: ignore
            _, hidden_states_t = self.parameter.q_target(burnin_state, hidden_states_t)  # type: ignore

        _, _, td_error, _, loss = self._train_steps(
            step_states_list,
            step_actions_list,
            step_actions_onehot_list,
            step_mu_probs_list,
            step_rewards_list,
            step_dones_list,
            step_next_invalid_actions_list,
            weights,
            hidden_states,
            hidden_states_t,
            0,
        )
        return td_error, loss

    # Q値(LSTM hidden states)の予測はforward、td_error,retraceはbackwardで予測する必要あり
    def _train_steps(
        self,
        step_states_list,
        step_actions_list,
        step_actions_onehot_list,
        step_mu_probs_list,
        step_rewards_list,
        step_dones_list,
        step_next_invalid_actions_list,
        weights,
        hidden_states,
        hidden_states_t,
        idx,
    ):
        # 最後
        if idx == self.config.sequence_length:
            n_states = step_states_list[idx]
            n_q, _ = self.parameter.q_online(n_states, hidden_states)  # type:ignore , ignore check "None"
            n_q_target, _ = self.parameter.q_target(n_states, hidden_states_t)  # type:ignore , ignore check "None"
            n_q = tf.stop_gradient(n_q).numpy()  # type:ignore , ignore check "None"
            n_q_target = tf.stop_gradient(n_q_target).numpy()  # type:ignore , ignore check "None"
            # q, target_q, td_error, retrace, loss
            return n_q, n_q_target, 0.0, 1.0, 0.0

        states = step_states_list[idx]

        # return用にq_targetを計算
        q_target, n_hidden_states_t = self.parameter.q_target(
            states, hidden_states_t
        )  # type:ignore , ignore check "None"
        q_target = tf.stop_gradient(q_target).numpy()  # type:ignore , ignore check "None"

        # --- 勾配 + targetQを計算
        with tf.GradientTape() as tape:
            q, n_hidden_states = self.parameter.q_online(states, hidden_states)  # type:ignore , ignore check "None"

            # 次のQ値を取得
            n_q, n_q_target, n_td_error, retrace, n_loss = self._train_steps(
                step_states_list,
                step_actions_list,
                step_actions_onehot_list,
                step_mu_probs_list,
                step_rewards_list,
                step_dones_list,
                step_next_invalid_actions_list,
                weights,
                n_hidden_states,
                n_hidden_states_t,
                idx + 1,
            )

            # targetQを計算
            target_q = np.zeros(self.config.batch_size)
            for i in range(self.config.batch_size):
                reward = step_rewards_list[idx][i]
                done = step_dones_list[idx][i]
                next_invalid_actions = step_next_invalid_actions_list[idx][i]

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
                    gain = reward + self.config.discount * maxq
                if self.config.enable_rescale:
                    gain = rescaling(gain)
                target_q[i] = gain

            if self.config.enable_retrace:
                _retrace = np.zeros(self.config.batch_size)
                for i in range(self.config.batch_size):
                    action = step_actions_list[idx][i]
                    mu_prob = step_mu_probs_list[idx][i]
                    pi_probs = calc_epsilon_greedy_probs(
                        n_q[i],
                        step_next_invalid_actions_list[idx][i],
                        0.0,
                        self.config.action_num,
                    )
                    pi_prob = pi_probs[action]
                    _retrace[i] = self.config.retrace_h * np.minimum(1, pi_prob / mu_prob)

                retrace *= _retrace
                target_q += self.config.discount * retrace * n_td_error

            action_onehot = step_actions_onehot_list[idx]
            q_onehot = tf.reduce_sum(q * action_onehot, axis=1)

            loss = self.loss(target_q * weights, q_onehot * weights)
            loss += tf.reduce_sum(self.parameter.q_online.losses)

        grads = tape.gradient(loss, self.parameter.q_online.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.parameter.q_online.trainable_variables))
        # --- 勾配計算ここまで
        q = tf.stop_gradient(q).numpy()  # type:ignore , ignore check "None"

        if idx == 0 or self.config.enable_retrace:
            td_error = target_q - q_onehot.numpy() + self.config.discount * retrace * n_td_error  # type: ignore
        else:
            td_error = 0
        return q, q_target, td_error, retrace, (loss.numpy() + n_loss) / 2


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(DiscreteActionWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter
        self.remote_memory: RemoteMemory = self.remote_memory

        self.dummy_state = np.full(self.config.observation_shape, self.config.dummy_state_val, dtype=np.float32)

    def call_on_reset(self, state: np.ndarray, invalid_actions: List[int]) -> dict:
        # states : burnin + sequence_length + next_state
        # actions: sequence_length
        # probs  : sequence_length
        # rewards: sequence_length
        # done   : sequence_length
        # invalid_actions: sequence_length + next_invalid_actions - now_invalid_actions
        # hidden_state   : burnin + sequence_length + next_state

        self._recent_states = [self.dummy_state for _ in range(self.config.burnin + self.config.sequence_length + 1)]
        self.recent_actions = [
            random.randint(0, self.config.action_num - 1) for _ in range(self.config.sequence_length)
        ]
        self.recent_probs = [1.0 / self.config.action_num for _ in range(self.config.sequence_length)]
        self.recent_rewards = [0.0 for _ in range(self.config.sequence_length)]
        self.recent_done = [False for _ in range(self.config.sequence_length)]
        self.recent_invalid_actions = [[] for _ in range(self.config.sequence_length)]

        self.hidden_state = self.parameter.q_online.init_hidden_state()
        self.recent_hidden_states = [
            [self.hidden_state[0][0], self.hidden_state[1][0]]
            for _ in range(self.config.burnin + self.config.sequence_length + 1)
        ]

        self._recent_states.pop(0)
        self._recent_states.append(state.astype(float))
        self.recent_invalid_actions.pop(0)
        self.recent_invalid_actions.append(invalid_actions)

        # TD誤差を計算するか
        if self.config.memory_name == "ReplayMemory":
            self._calc_td_error = False
        elif not self.distributed:
            self._calc_td_error = False
        else:
            self._calc_td_error = True
            self._history_batch = []

        return {}

    def call_policy(self, state: np.ndarray, invalid_actions: List[int]) -> Tuple[int, dict]:
        state = np.asarray([[state]] * self.config.batch_size)
        q, self.hidden_state = self.parameter.q_online(state, self.hidden_state)  # type:ignore , ignore check "None"
        q = q[0].numpy()

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
        self._recent_states.append(next_state.astype(float))
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

    def render_terminal(self, worker, **kwargs) -> None:
        # パラメータを予測するとhidden_stateが変わってしまうの予測はしない
        q = self.q
        maxa = np.argmax(q)
        if self.config.enable_rescale:
            q = inverse_rescaling(q)

        def _render_sub(a: int) -> str:
            return f"{q[a]:7.5f}"

        render_discrete_action(maxa, worker.env, self.config, _render_sub)

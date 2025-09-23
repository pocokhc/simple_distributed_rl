from typing import Any, cast

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.rl.algorithms.base_ppo import RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.trainer import RLTrainer
from srl.base.spaces.discrete import DiscreteSpace
from srl.rl.tf import helper as helper_tf
from srl.rl.tf.distributions.categorical_gumbel_dist_block import CategoricalGumbelDistBlock
from srl.rl.tf.model import KerasModelAddedSummary

from .config import Config, Memory

kl = keras.layers


class PolicyNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        act_space = cast(DiscreteSpace, config.action_space)

        self.in_block = config.input_block.create_tf_block(config)
        self.hidden_block = config.policy_block.create_tf_block()

        # out
        self.policy_dist_block = CategoricalGumbelDistBlock(act_space.n)

        # build
        self(config.input_block.create_tf_dummy_data(config))

    def call(self, x, training=False) -> Any:
        x = self.in_block(x, training=training)
        x = self.hidden_block(x, training=training)
        return self.policy_dist_block(x)

    # @tf.function
    def compute_train_loss(self, state, q_val, alpha):
        p_dist = self(state, training=True)
        probs = p_dist.probs()
        log_probs = tf.math.log(tf.clip_by_value(probs, 10e-8, 1))
        entropy = -tf.reduce_sum(probs * log_probs, axis=-1, keepdims=True)

        policy_loss = tf.reduce_sum(probs * (alpha * log_probs - q_val), axis=-1)
        policy_loss = tf.reduce_mean(policy_loss)

        policy_loss += tf.reduce_sum(self.losses)  # 正則化項
        return policy_loss, entropy


def compute_discrete_all_q(state, act_num: int, q1_model, q2_model):
    # 全てのアクションでQ値を計算
    batch_size = tf.shape(state)[0]
    action_ids = tf.range(act_num, dtype=tf.int32)
    state_expand = tf.repeat(state, repeats=act_num, axis=0)  # (B*A, state_dim)
    action_onehot = tf.tile(tf.one_hot(action_ids, depth=act_num, dtype=state.dtype), [batch_size, 1])  # (B*A, A)

    q1 = q1_model([state_expand, action_onehot])
    q2 = q2_model([state_expand, action_onehot])
    q_min = tf.minimum(q1, q2)

    q_min = tf.reshape(q_min, [batch_size, act_num])  # (B*A) -> (B, A)
    return q_min


class QNetwork(KerasModelAddedSummary):
    def __init__(self, config: Config):
        super().__init__()
        act_space = cast(DiscreteSpace, config.action_space)

        self.in_block = config.input_block.create_tf_block(config)
        self.q_block = config.q_block.create_tf_block()
        self.q_out_layer = kl.Dense(1)

        # build
        self(
            [
                config.input_block.create_tf_dummy_data(config),
                np.zeros((1, act_space.n)),
            ]
        )

    def call(self, x, training=False):
        state = x[0]
        action = x[1]

        state = self.in_block(state, training=training)
        x = tf.concat([state, action], axis=1)

        x = self.q_block(x, training=training)
        x = self.q_out_layer(x, training=training)
        return x

    @tf.function
    def compute_train_loss(self, state, action, target_q):
        q = self([state, action], training=True)
        loss = tf.reduce_mean(tf.square(target_q - q))
        loss += tf.reduce_sum(self.losses)  # 正則化項
        return loss


class Parameter(RLParameter):
    def setup(self):
        self.policy = PolicyNetwork(self.config)
        self.q1_online = QNetwork(self.config)
        self.q1_target = QNetwork(self.config)
        self.q1_target.set_weights(self.q1_online.get_weights())
        self.q2_online = QNetwork(self.config)
        self.q2_target = QNetwork(self.config)
        self.q2_target.set_weights(self.q2_online.get_weights())

        # エントロピーα自動調整用
        self.log_alpha = tf.Variable(np.log(self.config.entropy_alpha), dtype=tf.float32)
        self.load_log_alpha = False

    def call_restore(self, data: Any, **kwargs) -> None:
        self.policy.set_weights(data[0])
        self.q1_online.set_weights(data[1])
        self.q1_target.set_weights(data[1])
        self.q2_online.set_weights(data[2])
        self.q2_target.set_weights(data[2])
        self.log_alpha = data[3]
        self.load_log_alpha = False

    def call_backup(self, **kwargs) -> Any:
        return [
            self.policy.get_weights(),
            self.q1_online.get_weights(),
            self.q2_online.get_weights(),
            self.log_alpha,
        ]

    def summary(self, **kwargs):
        self.policy.summary(**kwargs)
        self.q1_online.summary(**kwargs)


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        self.act_space = cast(DiscreteSpace, self.config.action_space)
        self.np_dtype = self.config.get_dtype("np")
        self.q1_optimizer = keras.optimizers.Adam(learning_rate=self.config.lr_q_scheduler.apply_tf_scheduler(self.config.lr_q))
        self.q2_optimizer = keras.optimizers.Adam(learning_rate=self.config.lr_q_scheduler.apply_tf_scheduler(self.config.lr_q))
        self.policy_optimizer = keras.optimizers.Adam(learning_rate=self.config.lr_policy_scheduler.apply_tf_scheduler(self.config.lr_policy))
        self.alpha_optimizer = None
        self.parameter.load_log_alpha = False

        # エントロピーαの目標値、-1*アクション数が良いらしい
        if isinstance(self.config.action_space, DiscreteSpace):
            n = self.config.action_space.n
        else:
            n = self.config.action_space.size
        self.target_entropy = -1 * n

    def train(self) -> None:
        batches = self.memory.sample()
        if batches is None:
            return

        state, action, n_state, reward, done = zip(*batches)
        state = np.asarray(state, dtype=self.np_dtype)
        action = np.asarray(action, dtype=self.np_dtype)
        n_state = np.asarray(n_state, dtype=self.np_dtype)
        done = np.asarray(done, dtype=self.np_dtype)[..., np.newaxis]
        reward = np.asarray(reward, dtype=self.np_dtype)[..., np.newaxis]

        if not self.parameter.load_log_alpha:
            # restore時に再度作り直す必要あり
            self.alpha_optimizer = keras.optimizers.Adam(learning_rate=self.config.lr_alpha_scheduler.apply_tf_scheduler(self.config.lr_alpha))
            self.parameter.load_log_alpha = True

        # 方策エントロピーの反映率αを計算
        alpha = np.exp(self.parameter.log_alpha)

        # ポリシーより次の状態のアクションを取得し、次の状態のアクションlogpiを取得
        n_p_dist = self.parameter.policy(n_state)
        n_probs = n_p_dist.probs()
        n_log_probs = tf.math.log(n_probs)

        n_q_min = compute_discrete_all_q(
            n_state,
            self.act_space.n,
            self.parameter.q1_target,
            self.parameter.q2_target,
        )
        n_q_min = tf.reduce_sum(n_probs * (n_q_min - alpha * n_log_probs), axis=-1, keepdims=True)
        target_q = reward + (1 - done) * self.config.discount * n_q_min
        target_q = tf.stop_gradient(target_q)

        # --- Qモデルの学習
        with tf.GradientTape() as tape:
            q1_loss = self.parameter.q1_online.compute_train_loss(state, action, target_q)
            q2_loss = self.parameter.q2_online.compute_train_loss(state, action, target_q)
            loss = q1_loss + q2_loss
        variables = [
            self.parameter.q1_online.trainable_variables,
            self.parameter.q2_online.trainable_variables,
        ]
        grads = tape.gradient(loss, variables)
        self.q1_optimizer.apply_gradients(zip(grads[0], variables[0]))
        self.q2_optimizer.apply_gradients(zip(grads[1], variables[1]))
        self.info["q1_loss"] = q1_loss.numpy()
        self.info["q2_loss"] = q2_loss.numpy()

        # --- ポリシーの学習
        q_val = compute_discrete_all_q(
            state,
            self.act_space.n,
            self.parameter.q1_online,
            self.parameter.q2_online,
        )
        with tf.GradientTape() as tape:
            policy_loss, entropy = self.parameter.policy.compute_train_loss(
                state,
                tf.stop_gradient(q_val),
                alpha,
            )
        grads = tape.gradient(policy_loss, self.parameter.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(grads, self.parameter.policy.trainable_variables))
        self.info["policy_loss"] = policy_loss.numpy()

        # --- 方策エントロピーαの自動調整
        if self.config.entropy_alpha_auto_scale:
            with tf.GradientTape() as tape:
                entropy_diff = self.target_entropy - entropy
                log_alpha_loss = -tf.reduce_mean(tf.exp(self.parameter.log_alpha) * entropy_diff)
            grad = tape.gradient(log_alpha_loss, self.parameter.log_alpha)
            self.alpha_optimizer.apply_gradients([(grad, self.parameter.log_alpha)])  # type: ignore
            self.info["alpha_loss"] = log_alpha_loss.numpy()
            self.info["alpha"] = alpha

        # --- target sync
        if self.train_count % self.config.hard_target_update_interval == 0:
            self.parameter.q1_target.set_weights(self.parameter.q1_online.get_weights())
            self.parameter.q2_target.set_weights(self.parameter.q2_online.get_weights())
        else:
            helper_tf.model_soft_sync(self.parameter.q1_target, self.parameter.q1_online, self.config.soft_target_update_tau)
            helper_tf.model_soft_sync(self.parameter.q2_target, self.parameter.q2_online, self.config.soft_target_update_tau)

        self.train_count += 1


class Worker(RLWorker[Config, Parameter, Memory]):
    def policy(self, worker):
        if self.training and self.step_in_training < self.config.start_steps:
            env_action = self.sample_action()
            self.action = env_action
            self.action = worker.get_onehot_action(env_action)
            return env_action

        p_dist = self.parameter.policy(worker.state[np.newaxis, ...])
        env_action = int(p_dist.sample().numpy()[0])
        self.action = worker.get_onehot_action(env_action)
        if self.rendering:
            self.probs = p_dist.probs().numpy()[0]
        return env_action

    def on_step(self, worker):
        if not self.training:
            return
        self.memory.add(
            [
                worker.state,  # state
                self.action,  # action
                worker.next_state,  # next state
                worker.reward,  # reward
                worker.terminated,  # terminate
            ]
        )

    def render_terminal(self, worker, **kwargs) -> None:
        act_space = cast(DiscreteSpace, self.config.action_space)
        state = worker.state[np.newaxis, ...]
        maxa = np.argmax(self.probs)

        def _render_sub(a: int) -> str:
            onehot_a = np.identity(act_space.n, dtype=np.float32)[a][np.newaxis, ...]
            q1 = self.parameter.q1_online([state, onehot_a])
            q2 = self.parameter.q2_online([state, onehot_a])
            q1 = q1.numpy()[0][0]
            q2 = q2.numpy()[0][0]

            s = f"{self.probs[a] * 100:5.1f}%, q1 {q1:7.4f}, q2 {q2:7.4f} "
            return s

        worker.print_discrete_action_info(int(maxa), _render_sub)

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.rl.trainer import RLTrainer
from srl.base.spaces.discrete import DiscreteSpace

from .config import Config
from .memory import Memory
from .parameter import Parameter


def scale_gradient(tensor, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def cross_entropy_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-6, y_pred)  # log(0)回避用
    loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1)
    return loss


def consist_loss_func(f1: tf.Tensor, f2: tf.Tensor) -> tf.Tensor:
    """
    一貫性損失（TensorFlow版）
    L2ノルムで正規化後、コサイン類似度を負にした値を返す。
    """
    f1 = tf.math.l2_normalize(f1, axis=-1, epsilon=1e-5)  # Why not: 正規化しないと内積がスケール依存になる
    f2 = tf.math.l2_normalize(f2, axis=-1, epsilon=1e-5)
    loss = -tf.reduce_sum(f1 * f2, axis=-1)
    return tf.reduce_mean(loss, axis=[1, 2])


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        self.np_dtype = self.config.get_dtype("np")
        self.tf_dtype = self.config.get_dtype("tf")
        self.opts = [
            keras.optimizers.Adam(learning_rate=self.config.lr_scheduler.apply_tf_scheduler(self.config.lr))
            for _ in range(4)  #
        ]
        self.is_discrete = isinstance(self.config.action_space, DiscreteSpace)

    def train(self) -> None:
        batches = self.memory.sample_batch()
        if batches is None:
            return
        states_list, actions_list, value_prefix_list, policies_list, z_list, weights, update_args = batches

        with tf.GradientTape() as tape:
            v_loss, p_loss, r_loss, c_loss = self._compute_train_loss(states_list, actions_list, value_prefix_list, policies_list, z_list)
            loss = tf.reduce_mean(
                (
                    self.config.value_loss_coeff * v_loss  #
                    + self.config.policy_loss_coeff * p_loss  #
                    + self.config.reward_loss_coeff * r_loss  #
                    + self.config.consistency_loss_coeff * c_loss  #
                )
                * weights
            )
            # 正則化項
            loss += tf.reduce_sum(self.parameter.representation_net.losses)
            loss += tf.reduce_sum(self.parameter.prediction_net.losses)
            loss += tf.reduce_sum(self.parameter.dynamics_net.losses)
            if self.config.consistency_loss_coeff > 0:
                loss += tf.reduce_sum(self.parameter.projector_net.losses)
        variables = [
            self.parameter.representation_net.trainable_variables,
            self.parameter.prediction_net.trainable_variables,
            self.parameter.dynamics_net.trainable_variables,
        ]
        if self.config.consistency_loss_coeff > 0:
            variables.append(self.parameter.projector_net.trainable_variables)
        grads = tape.gradient(loss, variables)
        for i in range(len(variables)):
            grads[i], _ = tf.clip_by_global_norm(grads[i], clip_norm=self.config.max_grad_norm)
            self.opts[i].apply_gradients(zip(grads[i], variables[i]))

        self.train_count += 1
        self.info["loss"] = loss.numpy()
        self.info["value_loss"] = np.mean(v_loss)
        self.info["policy_loss"] = np.mean(p_loss)
        self.info["reward_loss"] = np.mean(r_loss)
        self.info["cons_loss"] = np.mean(c_loss)

        # memory update
        priorities = np.abs(v_loss.numpy())
        self.memory.update(update_args, priorities, self.train_count)

        # --- 正規化用Qを保存(parameterはtrainerからしか保存されない)
        # (remote_memory -> trainer -> parameter)
        q = self.memory.get_q()
        if q is not None:
            self.parameter.q_min = min(self.parameter.q_min, q[0])
            self.parameter.q_max = max(self.parameter.q_max, q[1])

        # parameter update
        if self.config.enable_reanalyze:
            self.memory.update_parameter(self.parameter.backup())

    @tf.function
    def _compute_train_loss(self, states_list, actions_list: list, value_prefix_list: list, policies_list: list, z_list: list):
        hc = self.parameter.dynamics_net.get_initial_state(self.config.batch_size)
        s_state = self.parameter.representation_net(states_list[0], training=True)

        # --- 1st step
        p_dist, v_pred = self.parameter.prediction_net(s_state, training=True)
        if self.is_discrete:
            p_loss = scale_gradient(cross_entropy_loss(policies_list[0], p_dist.probs()), 1.0)
            p_loss += scale_gradient(-p_dist.log_prob(actions_list[0], onehot=True, keepdims=False), 1.0)
        else:
            p_loss = scale_gradient(-p_dist.log_prob(actions_list[0]), 1.0)
        v_loss = scale_gradient(cross_entropy_loss(z_list[0], v_pred), 1.0)
        r_loss = 0
        c_loss = 0

        # --- unroll steps
        gradient_scale = 1 / self.config.unroll_steps
        for t in range(self.config.unroll_steps):
            s_state, value_prefix_pred, hc = self.parameter.dynamics_net([s_state, actions_list[t], hc], training=True)
            p_dist, v_pred = self.parameter.prediction_net(s_state, training=True)

            # --- self-supervised consistency loss
            if self.config.consistency_loss_coeff > 0:
                n_s_state = self.parameter.representation_net(states_list[t + 1])
                target_poj = self.parameter.projector_net.projection(n_s_state, training=False)
                pred_proj = self.parameter.projector_net(s_state, training=True)
                c_loss += scale_gradient(consist_loss_func(pred_proj, tf.stop_gradient(target_poj)), gradient_scale)

            if self.is_discrete:
                p_loss += scale_gradient(cross_entropy_loss(policies_list[t + 1], p_dist.probs()), gradient_scale)
                p_loss += scale_gradient(-p_dist.log_prob(actions_list[t + 1], onehot=True, keepdims=False), gradient_scale)
            else:
                p_loss += scale_gradient(-p_dist.log_prob(actions_list[t + 1]), gradient_scale)
            v_loss += scale_gradient(cross_entropy_loss(z_list[t + 1], v_pred), gradient_scale)
            r_loss += scale_gradient(cross_entropy_loss(value_prefix_list[t], value_prefix_pred), gradient_scale)

            s_state = scale_gradient(s_state, 0.5)

        v_loss /= self.config.unroll_steps + 1
        p_loss /= self.config.unroll_steps + 1
        r_loss /= self.config.unroll_steps
        c_loss /= self.config.unroll_steps
        return v_loss, p_loss, r_loss, c_loss

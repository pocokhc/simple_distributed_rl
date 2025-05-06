import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.rl.trainer import RLTrainer
from srl.rl.tf.distributions.categorical_dist_block import CategoricalDist

from .config import Config
from .memory import Memory
from .parameter import Parameter


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        self.tf_dtype = self.config.get_dtype("tf")
        self.opt_reward_end = keras.optimizers.AdamW(
            learning_rate=self.config.reward_end_cfg.lr,
            weight_decay=self.config.reward_end_cfg.weight_decay,
            epsilon=self.config.reward_end_cfg.eps,
            clipnorm=self.config.reward_end_cfg.max_grad_norm,
        )
        self.opt_actor_critic = keras.optimizers.AdamW(
            learning_rate=self.config.actor_critic_cfg.lr,
            weight_decay=self.config.actor_critic_cfg.weight_decay,
            epsilon=self.config.actor_critic_cfg.eps,
            clipnorm=self.config.actor_critic_cfg.max_grad_norm,
        )
        self.v_loss_func = keras.losses.Huber()
        self.act_loss_func = keras.losses.Huber()

    def train(self) -> None:
        f1 = f2 = f3 = False
        if self.config.train_diffusion:
            f1 = self._update_diffusion_model()
        if self.config.train_reward_end:
            f2 = self._update_reward_end()
        if self.config.train_actor_critic:
            f3 = self._update_actor_critic()
        if f1 or f2 or f3:
            self.train_count += 1

    def _update_diffusion_model(self) -> bool:
        # --- burninの時系列バッチを取得
        batches = self.memory.sample_diff()
        if batches is None:
            return False

        # (batch, steps, shape)
        obs = np.asarray([steps[-1][0] for steps in batches])
        recent_obs = np.asarray([[b[0] for b in steps[:-1]] for steps in batches])
        recent_act = np.asarray([[b[1] for b in steps[1:]] for steps in batches])

        loss = self.parameter.denoiser.update(obs, recent_obs, recent_act)
        self.info.update({"loss_diff": loss})
        return True

    def _update_reward_end(self) -> bool:
        # --- burnin+horizonの時系列バッチを取得
        batches = self.memory.sample_rewend()
        if batches is None:
            return False

        # (batch, steps, shape)
        all_obs = np.asarray([[b[0] for b in steps] for steps in batches])
        act = np.asarray([[b[1] for b in steps] for steps in batches])[:, 1:, ...]
        reward = np.asarray([[b[2] for b in steps] for steps in batches])[:, 1:, ...]
        done = np.asarray([[b[3] for b in steps] for steps in batches])[:, 1:, ...]

        obs = all_obs[:, :-1, ...]
        next_obs = all_obs[:, 1:, ...]

        # burnin
        hc_rewend = self.parameter.reward_end_model.get_initial_state(self.config.batch_size)
        if self.config.burnin > 0:
            _, _, hc_rewend = self.parameter.reward_end_model(
                [
                    obs[:, : self.config.burnin, ...],
                    act[:, : self.config.burnin, ...],
                    next_obs[:, : self.config.burnin, ...],
                ],
                hc=hc_rewend,
            )

        with tf.GradientTape() as tape:
            loss_r, loss_d, _ = self.parameter.reward_end_model.compute_train_loss(
                obs[:, self.config.burnin :, ...],
                act[:, self.config.burnin :, ...],
                next_obs[:, self.config.burnin :, ...],
                reward[:, self.config.burnin :, ...],
                done[:, self.config.burnin :, ...],
                hc_rewend,
            )
            loss = loss_r + loss_d
            loss += tf.reduce_sum(self.parameter.reward_end_model.losses)  # layer正則化項
        grad = tape.gradient(loss, self.parameter.reward_end_model.trainable_variables)
        self.opt_reward_end.apply_gradients(zip(grad, self.parameter.reward_end_model.trainable_variables))
        self.info.update(
            {
                "loss_r": loss_r.numpy(),
                "loss_d": loss_d.numpy(),
            }
        )

        return True

    def _update_actor_critic(self) -> bool:
        batches = self.memory.sample_actor_critic()
        if batches is None:
            return False

        # (batch, steps, shape)
        all_obs = np.asarray([[b[0] for b in steps] for steps in batches])
        acts = np.asarray([[b[1] for b in steps] for steps in batches])

        obs = all_obs[:, :-1, ...]
        next_obs = all_obs[:, 1:, ...]

        # burnin
        hc_rewend = self.parameter.reward_end_model.get_initial_state(self.config.batch_size)
        hc_act = self.parameter.actor_critic.get_initial_state(self.config.batch_size)
        if self.config.denoiser_cfg.num_steps_conditioning > 0:
            _, _, hc_rewend = self.parameter.reward_end_model(
                [
                    obs,
                    acts[:, 1:, ...],
                    next_obs,
                ],
                hc=hc_rewend,
            )
        for i in range(self.config.denoiser_cfg.num_steps_conditioning - 1):
            _, _, hc_act = self.parameter.actor_critic(obs[:, i, ...], hc=hc_act)

        with tf.GradientTape() as tape:
            loss_a, loss_v, loss_entropy = self._compute_actor_critic_loss(
                obs,
                acts[:, :-1, ...],
                hc_rewend,
                hc_act,
            )
            loss = loss_a + self.config.weight_value_loss * loss_v + self.config.weight_entropy_loss * loss_entropy
            loss += tf.reduce_sum(self.parameter.actor_critic.losses)  # layer正則化項
        grad = tape.gradient(loss, self.parameter.actor_critic.trainable_variables)
        self.opt_actor_critic.apply_gradients(zip(grad, self.parameter.actor_critic.trainable_variables))
        self.info.update(
            {
                "loss_a": loss_a.numpy(),
                "loss_v": loss_v.numpy(),
                "loss_entropy": loss_entropy.numpy(),
            }
        )
        return True

    def _compute_actor_critic_loss(self, recent_obs, recent_act, hc_rewend, hc_act):
        reward_list = []
        dones = []
        v_list = []
        logpi_list = []
        loss_entropy_list = []
        recent_obs = tf.convert_to_tensor(recent_obs, dtype=self.tf_dtype)
        r_weights = tf.constant([-1, 0, 1], dtype=self.tf_dtype)
        d_weights = tf.constant([0, 1], dtype=self.tf_dtype)
        for i in range(self.config.horizon):
            # predict action, value
            obs = recent_obs[:, -1, ...]
            act_dist, v, hc_act = self.parameter.actor_critic(obs, hc=hc_act, training=True)
            v_list.append(v)
            act_dist: CategoricalDist = act_dist  # 型明示
            act = act_dist.rsample()
            logpi_list.append(act_dist.log_prob(act, keepdims=True))
            argmax_act = tf.argmax(tf.expand_dims(act, axis=1), axis=-1)
            recent_act = tf.concat([recent_act, argmax_act], axis=1)
            recent_act = recent_act[:, 1:, ...]
            loss_entropy_list.append(act_dist.entropy())

            # sample next state
            next_obs, _ = self.parameter.sampler.sample(recent_obs, recent_act)
            next_obs = tf.stop_gradient(next_obs)
            next_obs = tf.expand_dims(next_obs, axis=1)
            recent_obs = tf.concat([recent_obs, next_obs], axis=1)
            recent_obs = recent_obs[:, 1:, ...]

            # sample reward, done(1step)
            obs = tf.expand_dims(obs, axis=1)
            r, d, hc_rewend = self.parameter.reward_end_model([obs, argmax_act, next_obs], hc=hc_rewend)
            d = tf.nn.softmax(d, axis=-1)
            d = tf.tensordot(d, d_weights, axes=[[-1], [0]]).numpy().astype(np.float32)
            dones.append(d)
            r = tf.nn.softmax(r, axis=-1)
            r = tf.tensordot(r, r_weights, axes=[[-1], [0]]).numpy().astype(np.float32)
            reward_list.append(r)

        gamma = self.config.discount
        lambda_ = self.config.lambda_

        # --- ラムダリターンの計算 (逆方向)
        # horizonの最後から逆方向に計算
        loss_a_list = []
        loss_v_list = []
        G_lambda = 0
        for i in reversed(range(self.config.horizon)):
            logpi = logpi_list[i]
            v = v_list[i]
            r = reward_list[i]
            d = dones[i]

            G_lambda = r + (1 - d) * gamma * ((1 - lambda_) * v + lambda_ * G_lambda)

            # loss a: logpi * A
            adv = G_lambda - v
            loss_a_list.append(tf.reduce_sum(logpi * tf.stop_gradient(adv), axis=-1))

            # loss v
            loss_v_list.append(self.v_loss_func(tf.stop_gradient(G_lambda), v))

        return (
            -tf.reduce_mean(loss_a_list),
            tf.reduce_mean(loss_v_list),
            -tf.reduce_mean(loss_entropy_list),
        )

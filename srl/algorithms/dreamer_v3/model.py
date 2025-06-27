from typing import Any, Tuple, cast

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

from srl.base.exception import UndefinedError
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.np_array import NpArraySpace
from srl.rl.tf.distributions.categorical_dist_block import CategoricalDist
from srl.rl.tf.distributions.linear_block import LinearBlock
from srl.rl.tf.distributions.normal_dist_block import NormalDistBlock
from srl.rl.tf.model import KerasModelAddedSummary
from srl.utils.common import compare_less_version

from .config import Config

kl = keras.layers
tfd = tfp.distributions
v216_older = compare_less_version(tf.__version__, "2.16.0")


class RSSM(KerasModelAddedSummary):
    def __init__(
        self,
        deter: int,
        stoch: int,
        classes: int,
        hidden_units: int,
        unimix: float,
        activation: Any,
        use_norm_layer: bool,
        use_categorical_distribution: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_categorical_distribution = use_categorical_distribution
        self.stoch_size = stoch
        self.classes = classes
        self.unimix = unimix

        # --- img step
        self.img_in_layers = [kl.Dense(hidden_units)]
        if use_norm_layer:
            self.img_in_layers.append(kl.LayerNormalization())
        self.img_in_layers.append(kl.Activation(activation))
        self.gru_cell = kl.GRUCell(deter)
        self.img_out_layers = [kl.Dense(hidden_units)]
        if use_norm_layer:
            self.img_out_layers.append(kl.LayerNormalization())
        self.img_out_layers.append(kl.Activation(activation))

        # --- obs step
        self.obs_layers = [kl.Dense(hidden_units)]
        if use_norm_layer:
            self.obs_layers.append(kl.LayerNormalization())
        self.obs_layers.append(kl.Activation(activation))

        self.concat_layer = kl.Concatenate(axis=-1)

        # --- dist
        if self.use_categorical_distribution:
            self.img_cat_dist_layers = [
                kl.Dense(stoch * classes, kernel_initializer="zeros"),
                kl.Reshape((stoch, classes)),
            ]
            self.obs_cat_dist_layers = [
                kl.Dense(stoch * classes, kernel_initializer="zeros"),
                kl.Reshape((stoch, classes)),
            ]
        else:
            self.img_norm_dist_block = NormalDistBlock(stoch * classes, (), (), ())
            self.obs_norm_dist_block = NormalDistBlock(stoch * classes, (), (), ())

    def img_step(self, prev_stoch, prev_deter, prev_onehot_action, training: bool = False):
        # --- NN
        x = tf.concat([prev_stoch, prev_onehot_action], -1)
        for layer in self.img_in_layers:
            x = layer(x, training=training)
        x, deter = self.gru_cell(x, [prev_deter], training=training)
        deter = deter[0]
        for layer in self.img_out_layers:
            x = layer(x, training=training)

        # --- dist
        if self.use_categorical_distribution:
            for h in self.img_cat_dist_layers:
                x = h(x)
            # (batch, stoch, classes) -> (batch * stoch, classes)
            batch = x.shape[0]
            x = tf.reshape(x, (batch * self.stoch_size, self.classes))
            dist = CategoricalDist(x).to_unimix_dist(self.unimix)
            # (batch * stoch, classes) -> (batch, stoch, classes) -> (batch, stoch * classes)
            stoch = tf.cast(
                tf.reshape(dist.rsample(), (batch, self.stoch_size, self.classes)),
                tf.float32,
            )
            stoch = tf.reshape(stoch, (batch, self.stoch_size * self.classes))
            # (batch * stoch, classes)
            probs = dist.probs()
            prior = {"stoch": stoch, "probs": probs}
        else:
            dist = self.img_norm_dist_block(x)
            prior = {
                "stoch": dist.rsample(),
                "mean": dist.mean(),
                "stddev": dist.stddev(),
            }

        return deter, prior

    def obs_step(self, deter, embed, training=False):
        # --- NN
        x = tf.concat([deter, embed], -1)
        for layer in self.obs_layers:
            x = layer(x, training=training)

        # --- dist
        if self.use_categorical_distribution:
            for h in self.obs_cat_dist_layers:
                x = h(x)
            # (batch, stoch, classes) -> (batch * stoch, classes)
            batch = x.shape[0]
            x = tf.reshape(x, (batch * self.stoch_size, self.classes))
            dist = CategoricalDist(x).to_unimix_dist(self.unimix)
            # (batch * stoch, classes) -> (batch, stoch, classes) -> (batch, stoch * classes)
            stoch = tf.cast(
                tf.reshape(dist.rsample(), (batch, self.stoch_size, self.classes)),
                tf.float32,
            )
            stoch = tf.reshape(stoch, (batch, self.stoch_size * self.classes))
            # (batch * stoch, classes)
            probs = dist.probs()
            post = {"stoch": stoch, "probs": probs}
        else:
            dist = self.obs_norm_dist_block(x)
            post = {
                "stoch": dist.rsample(),
                "mean": dist.mean(),
                "stddev": dist.stddev(),
            }

        return post

    def get_initial_state(self, batch_size: int = 1):
        stoch = tf.zeros((batch_size, self.stoch_size * self.classes), dtype=self.dtype)
        if v216_older:
            deter = self.gru_cell.get_initial_state(None, batch_size, dtype=self.dtype)
        else:
            deter = self.gru_cell.get_initial_state(batch_size)[0]
        return stoch, deter

    @tf.function
    def compute_train_loss(self, embed, actions, stoch, deter, undone, batch_size, batch_length, free_nats):
        # (seq*batch, shape) -> (seq, batch, shape)
        embed = tf.reshape(embed, (batch_length, batch_size) + embed.shape[1:])
        undone = tf.reshape(undone, (batch_length, batch_size) + undone.shape[1:])

        # --- batch seq step
        stochs = []
        deters = []
        if self.use_categorical_distribution:
            post_probs = []
            prior_probs = []
            for i in range(batch_length):
                deter, prior = self.img_step(stoch, deter, actions[i], training=True)
                post = self.obs_step(deter, embed[i], training=True)
                stoch = post["stoch"]
                stochs.append(stoch)
                deters.append(deter)
                post_probs.append(post["probs"])
                prior_probs.append(prior["probs"])
                # 終了時は初期化
                stoch = stoch * undone[i]
                deter = deter * undone[i]
            post_probs = tf.stack(post_probs, axis=0)
            prior_probs = tf.stack(prior_probs, axis=0)

            # 多分KLの計算でlogが使われるので確率0があるとinfになる
            post_probs = tf.clip_by_value(post_probs, 1e-10, 1)  # log(0)回避用
            prior_probs = tf.clip_by_value(prior_probs, 1e-10, 1)  # log(0)回避用
            post_dist = tfd.OneHotCategorical(probs=post_probs)
            prior_dist = tfd.OneHotCategorical(probs=prior_probs)

        else:
            post_mean = []
            post_std = []
            prior_mean = []
            prior_std = []
            for i in range(batch_length):
                deter, prior = self.img_step(stoch, deter, actions[i], training=True)
                post = self.obs_step(deter, embed[i], training=True)
                stoch = post["stoch"]
                stochs.append(stoch)
                deters.append(deter)
                post_mean.append(post["mean"])
                post_std.append(post["stddev"])
                prior_mean.append(prior["mean"])
                prior_std.append(prior["stddev"])
                # 終了時は初期化
                stoch = stoch * undone[i]
                deter = deter * undone[i]

            post_mean = tf.stack(post_mean, axis=0)
            post_std = tf.stack(post_std, axis=0)
            prior_mean = tf.stack(prior_mean, axis=0)
            prior_std = tf.stack(prior_std, axis=0)

            post_dist = tfd.Normal(post_mean, post_std)
            prior_dist = tfd.Normal(prior_mean, prior_std)

        stochs = tf.stack(stochs, axis=0)
        deters = tf.stack(deters, axis=0)

        # (seq, batch, shape) -> (seq*batch, shape)
        stochs = tf.reshape(stochs, (batch_length * batch_size,) + stochs.shape[2:])
        deters = tf.reshape(deters, (batch_length * batch_size,) + deters.shape[2:])
        feats = self.concat_layer([stochs, deters])

        # --- KL loss
        kl_loss_dyn = tfd.kl_divergence(tf.stop_gradient(post_dist), prior_dist)
        kl_loss_rep = tfd.kl_divergence(post_dist, tf.stop_gradient(prior_dist))
        kl_loss_dyn = tf.reduce_mean(tf.maximum(kl_loss_dyn, free_nats))
        kl_loss_rep = tf.reduce_mean(tf.maximum(kl_loss_rep, free_nats))

        return stochs, deters, feats, kl_loss_dyn, kl_loss_rep, stoch, deter

    def build_call(self, config: Config, embed_size: int):
        self._embed_size = embed_size
        in_stoch, in_deter = self.get_initial_state()
        if isinstance(config.action_space, DiscreteSpace):
            n = config.action_space.n
        elif isinstance(config.action_space, NpArraySpace):
            n = config.action_space.size
        in_onehot_action = np.zeros((1, n), dtype=np.float32)
        in_embed = np.zeros((1, embed_size), dtype=np.float32)
        deter, prior = self.img_step(in_stoch, in_deter, in_onehot_action)
        post = self.obs_step(deter, in_embed)
        return self.concat_layer([post["stoch"], deter])


class ImageEncoder(KerasModelAddedSummary):
    def __init__(
        self,
        img_shape: tuple,
        depth: int,
        res_blocks: int,
        activation,
        normalization_type: str,
        resize_type: str,
        resized_image_size: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert normalization_type in ["none", "layer"]
        self._in_shape = img_shape
        self.img_shape = img_shape

        _size = int(np.log2(min(img_shape[-3], img_shape[-2])))
        _resize = int(np.log2(resized_image_size))
        assert _size > _resize
        self.stages = _size - _resize

        if resize_type == "stride":
            assert img_shape[-2] % (2**self.stages) == 0
            assert img_shape[-3] % (2**self.stages) == 0
        elif resize_type == "stride3":
            assert (img_shape[-2] % ((2 ** (self.stages - 1)) * 3)) == 0
            assert (img_shape[-3] % ((2 ** (self.stages - 1)) * 3)) == 0
        elif resize_type == "max":
            assert img_shape[-2] % (2**self.stages) == 0
            assert img_shape[-3] % (2**self.stages) == 0
        else:
            raise NotImplementedError(resize_type)

        _conv_kw: dict = dict(
            padding="same",
            kernel_initializer=tf.initializers.TruncatedNormal(),
            bias_initializer="zero",
        )

        self.blocks = []
        for i in range(self.stages):
            # --- cnn
            use_bias = normalization_type == "none"
            if resize_type == "stride":
                cnn_layers = [kl.Conv2D(depth, 4, 2, use_bias=use_bias, **_conv_kw)]
            elif resize_type == "stride3":
                s = 2 if i else 3
                k = 5 if i else 4
                cnn_layers = [kl.Conv2D(depth, k, s, use_bias=use_bias, **_conv_kw)]
            elif resize_type == "mean":
                cnn_layers = [
                    kl.Conv2D(depth, 3, 1, use_bias=use_bias, **_conv_kw),
                    kl.AveragePooling2D((3, 3), (2, 2), padding="same"),
                ]
            elif resize_type == "max":
                cnn_layers = [
                    kl.Conv2D(depth, 3, 1, use_bias=use_bias, **_conv_kw),
                    kl.MaxPooling2D((3, 3), (2, 2), padding="same"),
                ]
            else:
                raise NotImplementedError(resize_type)
            if normalization_type == "layer":
                cnn_layers.append(kl.LayerNormalization())
            cnn_layers.append(kl.Activation(activation))

            # --- res
            res_blocks_layers = []
            for _ in range(res_blocks):
                res_layers = []
                if normalization_type == "layer":
                    res_layers.append(kl.LayerNormalization())
                res_layers.append(kl.Activation(activation))
                res_layers.append(kl.Conv2D(depth, 3, 1, use_bias=True, **_conv_kw))
                if normalization_type == "layer":
                    res_layers.append(kl.LayerNormalization())
                res_layers.append(kl.Activation(activation))
                res_layers.append(kl.Conv2D(depth, 3, 1, use_bias=True, **_conv_kw))
                res_blocks_layers.append(res_layers)

            self.blocks.append([cnn_layers, res_blocks_layers])
            depth *= 2

        self.out_layers = []
        if res_blocks > 0:
            self.out_layers.append(kl.Activation(activation))
        self.out_layers.append(kl.Flatten())

        dummy, img_shape = self._call(np.zeros((1,) + img_shape), return_size=True)
        self.resized_img_shape = img_shape[1:]
        self.out_size = dummy.shape[1]

    @tf.function
    def call(self, x, training=False):
        return self._call(x, training=training)

    def _call(self, x, training=False, return_size=False):
        x = x - 0.5
        for block in self.blocks:
            # --- cnn
            for h in block[0]:
                x = h(x, training=training)
            # --- res
            for res_blocks in block[1]:
                skip = x
                for h in res_blocks:
                    x = h(x, training=training)
                x += skip

        x_out = x
        for h in self.out_layers:
            x_out = h(x_out, training=training)

        if return_size:
            return x_out, x.shape
        else:
            return x_out


class ImageDecoder(KerasModelAddedSummary):
    def __init__(
        self,
        encoder: ImageEncoder,
        use_sigmoid: bool,
        depth: int,
        res_blocks: int,
        activation,
        normalization_type: str,
        resize_type: str,
        dist_type: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_sigmoid = use_sigmoid
        self.dist_type = dist_type

        stages = encoder.stages
        depth = depth * 2 ** (encoder.stages - 1)
        img_shape = encoder.img_shape
        resized_img_shape = encoder.resized_img_shape

        # --- in layers
        self.in_layer = kl.Dense(resized_img_shape[0] * resized_img_shape[1] * resized_img_shape[2])
        self.reshape_layer = kl.Reshape([resized_img_shape[0], resized_img_shape[1], resized_img_shape[2]])

        # --- conv layers
        _conv_kw: dict = dict(
            kernel_initializer=tf.initializers.TruncatedNormal(),
            bias_initializer="zero",
        )
        self.blocks = []
        for i in range(encoder.stages):
            # --- res
            res_blocks_layers = []
            for _ in range(res_blocks):
                res_layers = []
                if normalization_type == "layer":
                    res_layers.append(kl.LayerNormalization())
                res_layers.append(kl.Activation(activation))
                res_layers.append(kl.Conv2D(depth, 3, 1, padding="same", **_conv_kw))
                if normalization_type == "layer":
                    res_layers.append(kl.LayerNormalization())
                res_layers.append(kl.Activation(activation))
                res_layers.append(kl.Conv2D(depth, 3, 1, padding="same", **_conv_kw))
                res_blocks_layers.append(res_layers)

            if i == stages - 1:
                depth = img_shape[-1]
            else:
                depth //= 2

            # --- cnn
            use_bias = normalization_type == "none"
            if resize_type == "stride":
                cnn_layers = [kl.Conv2DTranspose(depth, 4, 2, use_bias=use_bias, padding="same", **_conv_kw)]
            elif resize_type == "stride3":
                s = 3 if i == stages - 1 else 2
                k = 5 if i == stages - 1 else 4
                cnn_layers = [kl.Conv2DTranspose(depth, k, s, use_bias=use_bias, padding="same", **_conv_kw)]
            elif resize_type == "max":
                cnn_layers = [
                    kl.UpSampling2D((2, 2)),
                    kl.Conv2D(depth, 3, 1, use_bias=use_bias, padding="same", **_conv_kw),
                ]
            else:
                raise NotImplementedError(resize_type)
            if normalization_type == "layer":
                cnn_layers.append(kl.LayerNormalization())
            cnn_layers.append(kl.Activation(activation))

            self.blocks.append([res_blocks_layers, cnn_layers])

        if dist_type == "linear":
            self.out_dist = LinearBlock(depth)
        elif dist_type == "normal":
            self.out_dist = NormalDistBlock(depth)
        else:
            raise UndefinedError(dist_type)

    def call(self, x):
        x = self.in_layer(x)
        x = self.reshape_layer(x)

        for block in self.blocks:
            # --- res
            for res_blocks in block[0]:
                skip = x
                for h in res_blocks:
                    x = h(x)
                x += cast(Any, skip)
            # --- cnn
            for h in block[1]:
                x = h(x)

        if self.use_sigmoid:
            x = tf.nn.sigmoid(x)
        else:
            x = cast(Any, x) + 0.5

        return self.out_dist(x)

    @tf.function
    def compute_train_loss(self, feat, state):
        dist = self(feat)
        if self.dist_type == "linear":
            return tf.reduce_mean(tf.square(state - dist.y))
        elif self.dist_type == "normal":
            return -tf.reduce_mean(dist.log_prob(state))
        else:
            raise UndefinedError(self.dist_type)


class LinearEncoder(KerasModelAddedSummary):
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...],
        activation: str,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_layers = []
        for size in hidden_layer_sizes:
            self.hidden_layers.append(kl.Dense(size, activation=activation))
        self.out_size: int = hidden_layer_sizes[-1]

    @tf.function
    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return x

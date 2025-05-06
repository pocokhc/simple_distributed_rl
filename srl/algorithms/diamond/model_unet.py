from functools import partial
from typing import List, Optional

import tensorflow as tf
from tensorflow import keras

from srl.rl.tf.model import KerasModelAddedSummary

kl = keras.layers

Conv2D1x1 = partial(kl.Conv2D, kernel_size=1, strides=1, padding="valid")
Conv2D3x3 = partial(kl.Conv2D, kernel_size=3, strides=1, padding="same")
IdentityLayer = partial(kl.Lambda, function=lambda x: x)  # 何もしないレイヤー


class GroupNorm(keras.layers.Layer):
    """group_sizeの調整を追加したGroupNorm"""

    def __init__(self, group_size: int = 32, eps: float = 1e-5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.group_size = group_size
        self.eps = eps

    def build(self, input_shape):
        in_channels = input_shape[-1]

        # group_sizeは割り切れる場合のみ指定、-1: LayerNorm, 1: InstanceNorm
        groups = self.group_size if in_channels % self.group_size == 0 else -1

        self.norm = kl.GroupNormalization(groups=groups, epsilon=self.eps)
        self.built = True

    def call(self, x, training=False):
        return self.norm(x, training=training)


class AdaGroupNorm2D(keras.layers.Layer):
    def __init__(self, group_size: int = 32, eps: float = 1e-5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.group_size = group_size
        self.eps = eps

    def build(self, input_shape):
        in_channels = input_shape[0][-1]
        self.norm = GroupNorm(self.group_size, self.eps)
        self.gamma = kl.Dense(in_channels, use_bias=False, kernel_initializer="zeros")
        self.beta = kl.Dense(in_channels, use_bias=False, kernel_initializer="zeros")
        self.built = True

    def call(self, inputs, training=False):
        x, condition = inputs
        x = self.norm(x, training=training)
        condition = tf.expand_dims(tf.expand_dims(condition, axis=1), axis=2)  # (b,c)->(b,1,1,c)
        gamma = self.gamma(condition)
        beta = self.beta(condition)
        return x * (1 + gamma) + beta


class SelfAttention2D(keras.layers.Layer):
    def __init__(self, head_dim: int = 8, **kwargs) -> None:
        super().__init__(**kwargs)
        self.head_dim = head_dim

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.n_head = max(1, in_channels // self.head_dim)
        assert in_channels % self.n_head == 0, f"Number that can be divided by the number of heads (head={self.n_head})"
        self.norm = GroupNorm()
        self.qkv_proj = Conv2D1x1(in_channels * 3)
        self.out_proj = Conv2D1x1(in_channels, kernel_initializer="zeros", bias_initializer="zeros")
        self.softmax = kl.Softmax(axis=-1)
        self.built = True

    def call(self, x, training=False):
        n, h, w, c = x.shape
        x = self.norm(x, training=training)
        qkv = self.qkv_proj(x)
        qkv = tf.reshape(qkv, (-1, h * w, c // self.n_head, self.n_head * 3))
        qkv = tf.transpose(qkv, perm=[0, 2, 1, 3])
        q, k, v = tf.split(qkv, num_or_size_splits=3, axis=-1)
        attn = tf.matmul(q, k, transpose_b=True)  # q@k.T
        attn = attn / tf.math.sqrt(tf.cast(k.shape[-1], x.dtype))
        attn = tf.matmul(self.softmax(attn), v)
        y = tf.transpose(attn, perm=[0, 2, 1, 3])
        y = tf.reshape(y, (-1, h, w, c))
        return x + self.out_proj(y)


class Downsample(keras.layers.Layer):
    def build(self, input_shape):
        self.conv = kl.Conv2D(
            filters=input_shape[-1],
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer=keras.initializers.Orthogonal(),
        )
        self.built = True

    def call(self, x, training=False):
        return self.conv(x, training=training)


class Upsample(keras.layers.Layer):
    def build(self, input_shape):
        self.conv = kl.Conv2D(
            filters=input_shape[-1],
            kernel_size=3,
            strides=1,
            padding="same",
        )
        self.built = True

    def call(self, x, training=False):
        input_shape = tf.shape(x)
        x = tf.image.resize(x, size=(input_shape[1] * 2, input_shape[2] * 2), method="nearest")
        return self.conv(x, training=training)


class SmallResBlock(KerasModelAddedSummary):
    def __init__(self, channels: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        use_projection = input_shape[-1] != self.channels
        self.proj = Conv2D1x1(self.channels) if use_projection else IdentityLayer()
        self.norm = GroupNorm()
        self.act = kl.Activation("silu")
        self.conv = Conv2D3x3(self.channels)
        self.built = True

    def call(self, x, training=False):
        r = self.proj(x, training=training)
        x = self.norm(x, training=training)
        x = self.act(x, training=training)
        x = self.conv(x, training=training)
        return x + r


class ResBlock(KerasModelAddedSummary):
    def __init__(self, channels: int, use_attention: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.channels = channels
        self.use_attention = use_attention

    def build(self, input_shape):
        use_projection = input_shape[0][-1] != self.channels
        self.proj = Conv2D1x1(self.channels) if use_projection else IdentityLayer()
        self.norm1 = AdaGroupNorm2D()
        self.act1 = kl.Activation("silu")
        self.conv1 = Conv2D3x3(self.channels)
        self.norm2 = AdaGroupNorm2D()
        self.act2 = kl.Activation("silu")
        self.conv2 = Conv2D3x3(self.channels)
        self.attn = SelfAttention2D() if self.use_attention else IdentityLayer()
        self.built = True

    def call(self, inputs, training=False):
        x, condition = inputs
        r = self.proj(x, training=training)
        x = self.norm1([x, condition], training=training)
        x = self.act1(x, training=training)
        x = self.conv1(x, training=training)
        x = self.norm2([x, condition], training=training)
        x = self.act2(x, training=training)
        x = self.conv2(x, training=training)
        x = x + r
        x = self.attn(x, training=training)
        return x


class ResBlocks(KerasModelAddedSummary):
    def __init__(self, channels_list: List[int], use_attention: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.resblocks = [ResBlock(c, use_attention) for c in channels_list]

    def call(self, inputs, shortcut: Optional[list] = None, training=False):
        x, condition = inputs
        outputs = []
        for i, resblock in enumerate(self.resblocks):
            if shortcut is not None:
                x = tf.concat([x, shortcut[i]], axis=-1)
            x = resblock([x, condition], training=training)
            outputs.append(x)
        return x, outputs


class UNet(KerasModelAddedSummary):
    def __init__(self, channels_list: list[int], res_block_num_list: list[int], use_attention_list: list[bool], **kwargs) -> None:
        super().__init__(**kwargs)
        if not (len(channels_list) == len(res_block_num_list) and len(res_block_num_list) == len(use_attention_list)):
            raise ValueError(f"{len(channels_list)} == {len(res_block_num_list)} == {len(use_attention_list)}")

        block_num = len(channels_list)
        self._num_downsampling = len(channels_list) - 1

        # downsamples
        self.downsamples: List[Downsample] = []
        self.d_blocks: List[ResBlocks] = []
        for i in range(block_num):
            self.downsamples.append(
                Downsample() if i != 0 else IdentityLayer(),
            )
            self.d_blocks.append(
                ResBlocks(
                    channels_list=[channels_list[i]] * res_block_num_list[i],
                    use_attention=use_attention_list[i],
                )
            )

        # middle
        self.m_blocks = ResBlocks(
            channels_list=[channels_list[-1]] * 2,
            use_attention=True,
        )

        # upsamples
        self.upsamples: List[Upsample] = []
        self.u_blocks: List[ResBlocks] = []
        for i in reversed(range(block_num)):
            self.upsamples.append(
                Upsample() if i != block_num - 1 else IdentityLayer(),
            )
            self.u_blocks.append(
                ResBlocks(
                    channels_list=[channels_list[i]] * res_block_num_list[i],
                    use_attention=use_attention_list[i],
                ),
            )

    def call(self, inputs, training=False):
        x, condition = inputs

        # downsamples
        d_outputs = []
        for block, down in zip(self.d_blocks, self.downsamples):
            x_down = down(x, training=training)
            x, block_outputs = block([x_down, condition], training=training)
            d_outputs.append((x_down, *block_outputs))

        # middle
        x, _ = self.m_blocks([x, condition], training=training)

        # upsamples
        u_outputs = []
        for block, up, skip in zip(self.u_blocks, self.upsamples, reversed(d_outputs)):
            x_up = up(x, training=training)
            x, block_outputs = block([x_up, condition], skip[::-1], training=training)
            u_outputs.append((x_up, *block_outputs))

        return x, d_outputs, u_outputs

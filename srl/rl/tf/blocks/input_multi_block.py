from typing import List

import numpy as np
import tensorflow as tf

from srl.base.spaces.box import BoxSpace
from srl.base.spaces.multi import MultiSpace
from srl.rl.models.config.input_image_block import InputImageBlockConfig
from srl.rl.models.config.input_value_block import InputValueBlockConfig
from srl.rl.tf.blocks.a_input_block import AInputBlock
from srl.rl.tf.model import KerasModelAddedSummary


class InputMultiBlockConcat(KerasModelAddedSummary, AInputBlock):
    def __init__(
        self,
        multi_space: MultiSpace,
        value_block_config: InputValueBlockConfig,
        image_block_config: InputImageBlockConfig,
        reshape_for_rnn: List[bool],
    ):
        super().__init__()
        self.reshape_for_rnn = reshape_for_rnn

        self.in_indices = []
        self.in_blocks = []
        self.in_shapes = []
        self.out_stypes = []
        for i, space in enumerate(multi_space.spaces):
            if not isinstance(space, BoxSpace):
                continue
            if space.is_value():
                self.in_blocks.append(value_block_config.create_tf_block(space, flatten=True))
            elif space.is_image():
                self.in_blocks.append(image_block_config.create_tf_block(space, out_flatten=True))
            else:
                continue
            self.in_indices.append(i)
            self.in_shapes.append(space.shape)
            self.out_stypes.append(space.stype)
        assert len(self.in_indices) > 0

    def call(self, x, training=False):
        x_arr = []
        i = -1
        for idx in self.in_indices:
            i += 1
            _x = x[idx]
            if self.reshape_for_rnn[idx]:
                size = tf.shape(_x)
                batch_size = size[0]
                seq_len = size[1]
                feat_dims = size[2:]
                _x = tf.reshape(_x, (batch_size * seq_len, *feat_dims))
            _x = self.in_blocks[i](_x, training=training)
            x_arr.append(_x)
        if self.concat:
            x_arr = tf.concat(x_arr, axis=-1)
        return x_arr

    # -----------------------

    def create_dummy_data(self, np_dtype, batch_size: int = 1, timesteps: int = 1) -> List[np.ndarray]:
        return [
            np.zeros(
                (batch_size, timesteps) + s if self.rnn[self.in_indices[i]] else (batch_size,) + s,
                np_dtype,
            )
            for i, s in enumerate(self.in_shapes)  #
        ]

    def to_tf_one_batch(self, data, tf_dtype, add_expand_dim: bool = True):
        if add_expand_dim:
            return [tf.cast(tf.expand_dims(data[i], axis=0), dtype=tf_dtype) for i in self.in_indices]
        else:
            return [tf.convert_to_tensor(data[i], dtype=tf_dtype) for i in self.in_indices]

    def to_tf_batches(self, data, tf_dtype):
        return [tf.convert_to_tensor(np.asarray([d[i] for d in data]), tf_dtype) for i in self.in_indices]

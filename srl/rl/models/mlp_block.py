from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class MLPBlockConfig:
    _kwargs: Dict[str, Any] = field(init=False, default_factory=lambda: {"layer_sizes": (512,)})

    def set_mlp(
        self,
        layer_sizes: Tuple[int, ...] = (512,),
        activation="relu",
        kernel_initializer="he_normal",
        dense_kwargs: Dict[str, Any] = {},
    ):
        self._kwargs = dict(
            layer_sizes=layer_sizes,
            activation=activation,
            kernel_initializer=kernel_initializer,
            dense_kwargs=dense_kwargs,
        )

    def set_original_block(self):
        raise NotImplementedError("TODO")

    # ---------------------

    def create_block_tf(self):
        from .mlp.tf import mlp_block

        return mlp_block.MLPBlock(
            **self._kwargs,
            enable_time_distributed_layer=False,
        )

    def create_block_torch(self, in_size: int):
        from .mlp.torch_ import mlp_block

        return mlp_block.MLPBlock(
            in_size,
            self._kwargs["layer_sizes"],
            # **self._kwargs,  TODO 互換パラメータの作成
        )

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DuelingNetworkConfig:
    name: str = field(default="")
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.name == "":
            self.set_dueling_network()

    def set(
        self,
        layer_sizes: Tuple[int, ...] = (512,),
        activation: str = "relu",
        **kwargs,
    ):
        """Multi-layer Perceptron Block

        Args:
            layer_sizes (Tuple[int, ...], optional): 各レイヤーのユニット数. Defaults to (512,).
            activation (str, optional): Activation function. Defaults to "relu".

        Examples:
            >>> mlp_conf = DuelingNetworkConfig()
            >>> mlp_conf.set((128, 64, 32))
        """
        self.name = "MLP"
        self.kwargs = dict(
            layer_sizes=layer_sizes,
            activation=activation.lower(),
        )
        self.kwargs.update(kwargs)
        return self

    def set_dueling_network(
        self,
        layer_sizes: Tuple[int, ...] = (512,),
        activation: str = "relu",
        dueling_type: str = "average",
        **kwargs,
    ):
        """Multi-layer Perceptron Block + DuelingNetwork Block

        Args:
            layer_sizes (Tuple[int, ...], optional): 各レイヤーのユニット数. Defaults to (512,).
            activation (str, optional): Activation function. Defaults to "relu".
            dueling_type (str, optional): select algorithm. Defaults to "average".

        """

        self.name = "DuelingNetwork"
        mlp_kwargs = dict(activation=activation.lower())
        mlp_kwargs.update(kwargs)
        self.kwargs = dict(
            layer_sizes=layer_sizes,
            mlp_kwargs=mlp_kwargs,
            dueling_kwargs=dict(
                dueling_type=dueling_type,
                activation=activation.lower(),
            ),
        )
        return self

    def set_custom_block(self, entry_point: str, kwargs: dict):
        self.name = "custom"
        self.kwargs = dict(entry_point=entry_point, kwargs=kwargs)
        return self

    # ---------------------

    def create_tf_block(self, out_size: int, rnn: bool = False, enable_noisy_dense: bool = False):
        from srl.rl.tf.blocks.dueling_network import create_block_from_config

        return create_block_from_config(self, out_size, rnn, enable_noisy_dense)

    def create_torch_block(self, in_size: int, out_size: int, enable_noisy_dense: bool = False):
        from srl.rl.torch_.blocks.dueling_network import create_block_from_config

        return create_block_from_config(self, in_size, out_size, enable_noisy_dense)

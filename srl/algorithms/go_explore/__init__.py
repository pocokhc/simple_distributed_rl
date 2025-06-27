from srl.base.rl.registration import register as _register

from .config import Config

_register(
    Config(),
    __name__ + ".go_explore:Memory",
    __name__ + ".go_explore:Parameter",
    __name__ + ".go_explore:Trainer",
    __name__ + ".go_explore:Worker",
)

# used Config class
from srl.rl.models.config.dueling_network import DuelingNetworkConfig  # noqa: F401, E402
from srl.rl.models.config.input_image_block import InputImageBlockConfig  # noqa: F401, E402
from srl.rl.models.config.input_value_block import InputValueBlockConfig  # noqa: F401, E402

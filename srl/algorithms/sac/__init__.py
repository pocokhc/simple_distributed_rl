from srl.base.rl.registration import register as _register

from .config import Config

_register(
    Config(),
    __name__ + ".sac:Memory",
    __name__ + ".sac:Parameter",
    __name__ + ".sac:Trainer",
    __name__ + ".sac:Worker",
)

# used Config class
from srl.rl.memories.replay_buffer import ReplayBufferConfig  # noqa: F401, E402
from srl.rl.models.config.input_image_block import InputImageBlockConfig  # noqa: F401, E402
from srl.rl.models.config.input_value_block import InputValueBlockConfig  # noqa: F401, E402
from srl.rl.models.config.mlp_block import MLPBlockConfig  # noqa: F401, E402
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig  # noqa: F401, E402

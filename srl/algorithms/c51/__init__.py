from srl.base.rl.registration import register as _register

from .config import Config

_register(
    Config(),
    __name__ + ".c51:Memory",
    __name__ + ".c51:Parameter",
    __name__ + ".c51:Trainer",
    __name__ + ".c51:Worker",
)

# used Config class
from srl.rl.memories.replay_buffer import ReplayBufferConfig  # noqa: F401, E402
from srl.rl.models.config.input_image_block import InputImageBlockConfig  # noqa: F401, E402
from srl.rl.models.config.input_value_block import InputValueBlockConfig  # noqa: F401, E402
from srl.rl.models.config.mlp_block import MLPBlockConfig  # noqa: F401, E402
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig  # noqa: F401, E402
from srl.rl.schedulers.scheduler import SchedulerConfig  # noqa: F401, E402

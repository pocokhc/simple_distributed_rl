from srl.base.rl.registration import register as _register

from .config import Config

_register(
    Config(),
    __name__ + ".r2d2:Memory",
    __name__ + ".r2d2:Parameter",
    __name__ + ".r2d2:Trainer",
    __name__ + ".r2d2:Worker",
)

# used Config class
from srl.rl.memories.priority_replay_buffer import PriorityReplayBufferConfig  # noqa: F401, E402
from srl.rl.models.config.dueling_network import DuelingNetworkConfig  # noqa: F401, E402
from srl.rl.models.config.input_image_block import InputImageBlockConfig  # noqa: F401, E402
from srl.rl.models.config.input_value_block import InputValueBlockConfig  # noqa: F401, E402
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig  # noqa: F401, E402

from srl.base.rl.registration import register as _register

from .agent57 import Config

_register(
    Config().set_tensorflow(),
    __name__ + ".agent57:Memory",
    __name__ + ".model_tf:Parameter",
    __name__ + ".model_tf:Trainer",
    __name__ + ".agent57:Worker",
)

_register(
    Config().set_torch(),
    __name__ + ".agent57:Memory",
    __name__ + ".model_torch:Parameter",
    __name__ + ".model_torch:Trainer",
    __name__ + ".agent57:Worker",
)

# used Config class
from srl.rl.memories.priority_replay_buffer import PriorityReplayBufferConfig  # noqa: F401, E402
from srl.rl.models.config.dueling_network import DuelingNetworkConfig  # noqa: F401, E402
from srl.rl.models.config.input_image_block import InputImageBlockConfig  # noqa: F401, E402
from srl.rl.models.config.mlp_block import MLPBlockConfig  # noqa: F401, E402
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig  # noqa: F401, E402
from srl.rl.schedulers.scheduler import SchedulerConfig  # noqa: F401, E402

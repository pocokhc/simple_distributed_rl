from srl.base.rl.registration import register as _register

from .config import Config

_register(
    Config(),
    __name__ + ".muzero:Memory",
    __name__ + ".muzero:Parameter",
    __name__ + ".muzero:Trainer",
    __name__ + ".muzero:Worker",
)

# used Config class
from srl.rl.memories.priority_replay_buffer import PriorityReplayBufferConfig  # noqa: F401, E402
from srl.rl.models.config.input_image_block import InputImageBlockConfig  # noqa: F401, E402
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig  # noqa: F401, E402
from srl.rl.schedulers.scheduler import SchedulerConfig  # noqa: F401, E402

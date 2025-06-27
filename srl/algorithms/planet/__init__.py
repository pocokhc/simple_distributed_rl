from srl.base.rl.registration import register as _register

from .config import Config

_register(
    Config(),
    __name__ + ".planet:Memory",
    __name__ + ".planet:Parameter",
    __name__ + ".planet:Trainer",
    __name__ + ".planet:Worker",
)

# used Config class
from srl.rl.memories.replay_buffer import ReplayBufferConfig  # noqa: F401, E402
from srl.rl.processors.image_processor import ImageProcessor  # noqa: F401, E402
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig  # noqa: F401, E402

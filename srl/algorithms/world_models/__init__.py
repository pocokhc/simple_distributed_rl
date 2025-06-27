from srl.base.rl.registration import register as _register

from .config import Config

_register(
    Config(),
    __name__ + ".world_models:Memory",
    __name__ + ".world_models:Parameter",
    __name__ + ".world_models:Trainer",
    __name__ + ".world_models:Worker",
)

# used Config class
from srl.rl.processors.image_processor import ImageProcessor  # noqa: F401, E402
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig  # noqa: F401, E402

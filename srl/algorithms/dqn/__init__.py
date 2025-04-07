from srl.base.rl.registration import register as _register

from .dqn import Config

_register(
    Config().set_tensorflow(),
    __name__ + ".dqn:Memory",
    __name__ + ".model_tf:Parameter",
    __name__ + ".model_tf:Trainer",
    __name__ + ".dqn:Worker",
)

_register(
    Config().set_torch(),
    __name__ + ".dqn:Memory",
    __name__ + ".model_torch:Parameter",
    __name__ + ".model_torch:Trainer",
    __name__ + ".dqn:Worker",
)

# used Config class
from srl.rl.memories.replay_buffer import ReplayBufferConfig  # noqa: F401, E402
from srl.rl.models.config.input_image_block import InputImageBlockConfig  # noqa: F401, E402
from srl.rl.models.config.mlp_block import MLPBlockConfig  # noqa: F401, E402
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig  # noqa: F401, E402
from srl.rl.schedulers.scheduler import SchedulerConfig  # noqa: F401, E402

from srl.base.rl.registration import register as _register

from .rainbow import Config

_register(
    Config(multisteps=3).set_tensorflow(),
    __name__ + ".rainbow:Memory",
    __name__ + ".model_tf:Parameter",
    __name__ + ".model_tf:Trainer",
    __name__ + ".rainbow:Worker",
)

_register(
    Config(multisteps=3).set_torch(),
    __name__ + ".rainbow:Memory",
    __name__ + ".model_torch:Parameter",
    __name__ + ".model_torch:Trainer",
    __name__ + ".rainbow:Worker",
)

_register(
    Config(multisteps=1).set_tensorflow(),
    __name__ + ".rainbow:Memory",
    __name__ + ".model_tf:Parameter",
    __name__ + ".model_tf:Trainer",
    __name__ + ".rainbow_nomultisteps:Worker",
)

_register(
    Config(multisteps=1).set_torch(),
    __name__ + ".rainbow:Memory",
    __name__ + ".model_torch:Parameter",
    __name__ + ".model_torch:Trainer",
    __name__ + ".rainbow_nomultisteps:Worker",
)

# used Config class
from srl.rl.memories.priority_replay_buffer import PriorityReplayBufferConfig  # noqa: F401, E402
from srl.rl.models.config.dueling_network import DuelingNetworkConfig  # noqa: F401, E402
from srl.rl.models.config.input_image_block import InputImageBlockConfig  # noqa: F401, E402
from srl.rl.models.config.mlp_block import MLPBlockConfig  # noqa: F401, E402
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig  # noqa: F401, E402
from srl.rl.schedulers.scheduler import SchedulerConfig  # noqa: F401, E402

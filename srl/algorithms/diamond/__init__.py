from srl.base.rl.registration import register

from .config import Config

register(
    Config(),
    __name__ + ".memory:Memory",
    __name__ + ".parameter:Parameter",
    __name__ + ".trainer:Trainer",
    __name__ + ".worker:Worker",
)

# used Config class
from .config import (  # noqa: E402
    ActorCriticConfig,  # noqa: F401, E402
    DenoiserConfig,  # noqa: F401, E402
    DiffusionSamplerConfig,  # noqa: F401, E402
    EpisodeReplayBufferConfig,  # noqa: F401, E402
    RewardEndModelConfig,  # noqa: F401, E402
)

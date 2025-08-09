from srl.base.rl.registration import register as _register

from .config import Config

_register(
    Config(),
    __name__ + ".sac:Memory",
    __name__ + ".sac:Parameter",
    __name__ + ".sac:Trainer",
    __name__ + ".sac:Worker",
)

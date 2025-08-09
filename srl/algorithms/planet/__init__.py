from srl.base.rl.registration import register as _register

from .config import Config

_register(
    Config(),
    __name__ + ".planet:Memory",
    __name__ + ".planet:Parameter",
    __name__ + ".planet:Trainer",
    __name__ + ".planet:Worker",
)

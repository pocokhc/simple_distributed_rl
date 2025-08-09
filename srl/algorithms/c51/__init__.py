from srl.base.rl.registration import register as _register

from .config import Config

_register(
    Config(),
    __name__ + ".c51:Memory",
    __name__ + ".c51:Parameter",
    __name__ + ".c51:Trainer",
    __name__ + ".c51:Worker",
)

from srl.base.rl.registration import register as _register

from .config import Config

_register(
    Config(),
    __name__ + ".r2d2:Memory",
    __name__ + ".r2d2:Parameter",
    __name__ + ".r2d2:Trainer",
    __name__ + ".r2d2:Worker",
)

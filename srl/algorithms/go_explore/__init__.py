from srl.base.rl.registration import register as _register

from .config import Config

_register(
    Config(),
    __name__ + ".go_explore:Memory",
    __name__ + ".go_explore:Parameter",
    __name__ + ".go_explore:Trainer",
    __name__ + ".go_explore:Worker",
)

from srl.base.rl.registration import register as _register

from .config import Config

_register(
    Config(),
    __name__ + ".muzero:Memory",
    __name__ + ".muzero:Parameter",
    __name__ + ".muzero:Trainer",
    __name__ + ".muzero:Worker",
)

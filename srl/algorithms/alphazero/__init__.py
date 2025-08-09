from srl.base.rl.registration import register as _register

from .config import Config

_register(
    Config(),
    __name__ + ".alphazero:Memory",
    __name__ + ".alphazero:Parameter",
    __name__ + ".alphazero:Trainer",
    __name__ + ".alphazero:Worker",
)

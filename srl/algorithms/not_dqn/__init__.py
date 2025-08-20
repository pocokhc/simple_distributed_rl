from srl.base.rl.registration import register as _register

from .config import Config

_register(
    Config(),
    __name__ + ".config:Memory",
    __name__ + ".parameter:Parameter",
    __name__ + ".trainer:Trainer",
    __name__ + ".worker:Worker",
)

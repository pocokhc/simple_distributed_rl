from srl.base.rl.registration import register as _register

from .config import Config

_register(
    Config(),
    __name__ + ".ddpg:Memory",
    __name__ + ".ddpg:Parameter",
    __name__ + ".ddpg:Trainer",
    __name__ + ".ddpg:Worker",
)

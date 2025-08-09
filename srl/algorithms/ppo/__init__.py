from srl.base.rl.registration import register as _register

from .config import Config

_register(
    Config(),
    __name__ + ".ppo:Memory",
    __name__ + ".ppo:Parameter",
    __name__ + ".ppo:Trainer",
    __name__ + ".ppo:Worker",
)

from srl.base.rl.registration import register as _register

from .config import Config

_register(
    Config(),
    __name__ + ".snd:Memory",
    __name__ + ".snd:Parameter",
    __name__ + ".snd:Trainer",
    __name__ + ".snd:Worker",
)

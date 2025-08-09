from srl.base.rl.registration import register as _register

from .config import Config

_register(
    Config(),
    __name__ + ".dreamer_v3:Memory",
    __name__ + ".dreamer_v3:Parameter",
    __name__ + ".dreamer_v3:Trainer",
    __name__ + ".dreamer_v3:Worker",
)

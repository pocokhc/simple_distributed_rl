from srl.base.rl.registration import register as _register

from .config import Config

_register(
    Config(),
    __name__ + ".go_dqn:Memory",
    __name__ + ".go_dqn:Parameter",
    __name__ + ".go_dqn:Trainer",
    __name__ + ".go_dqn:Worker",
)

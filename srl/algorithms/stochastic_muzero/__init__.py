from srl.base.rl.registration import register as _register

from .config import Config

_register(
    Config(),
    __name__ + ".stochastic_muzero:Memory",
    __name__ + ".stochastic_muzero:Parameter",
    __name__ + ".stochastic_muzero:Trainer",
    __name__ + ".stochastic_muzero:Worker",
)

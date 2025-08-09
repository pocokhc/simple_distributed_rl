from srl.base.rl.registration import register as _register

from .agent57 import Config

_register(
    Config().set_tensorflow(),
    __name__ + ".agent57:Memory",
    __name__ + ".model_tf:Parameter",
    __name__ + ".model_tf:Trainer",
    __name__ + ".agent57:Worker",
)

_register(
    Config().set_torch(),
    __name__ + ".agent57:Memory",
    __name__ + ".model_torch:Parameter",
    __name__ + ".model_torch:Trainer",
    __name__ + ".agent57:Worker",
)

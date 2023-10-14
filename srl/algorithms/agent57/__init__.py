from srl.base.rl.registration import register

from .agent57 import Config

_c = Config()
_c.framework.set_tensorflow()
register(
    _c,
    __name__ + ".agent57:Memory",
    __name__ + ".model_tf:Parameter",
    __name__ + ".model_tf:Trainer",
    __name__ + ".agent57:Worker",
)

_c = Config()
_c.framework.set_torch()
register(
    _c,
    __name__ + ".agent57:Memory",
    __name__ + ".model_torch:Parameter",
    __name__ + ".model_torch:Trainer",
    __name__ + ".agent57:Worker",
)

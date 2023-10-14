from srl.base.rl.registration import register

from .rainbow import Config

_c = Config(multisteps=3)
_c.framework.set_tensorflow()
register(
    _c,
    __name__ + ".rainbow:Memory",
    __name__ + ".model_tf:Parameter",
    __name__ + ".model_tf:Trainer",
    __name__ + ".rainbow:Worker",
)

_c = Config(multisteps=3)
_c.framework.set_torch()
register(
    _c,
    __name__ + ".rainbow:Memory",
    __name__ + ".model_torch:Parameter",
    __name__ + ".model_torch:Trainer",
    __name__ + ".rainbow:Worker",
)

_c = Config(multisteps=1)
_c.framework.set_tensorflow()
register(
    _c,
    __name__ + ".rainbow:Memory",
    __name__ + ".model_tf:Parameter",
    __name__ + ".model_tf:Trainer",
    __name__ + ".rainbow_nomultisteps:Worker",
)

_c = Config(multisteps=1)
_c.framework.set_torch()
register(
    _c,
    __name__ + ".rainbow:Memory",
    __name__ + ".model_torch:Parameter",
    __name__ + ".model_torch:Trainer",
    __name__ + ".rainbow_nomultisteps:Worker",
)

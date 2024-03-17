from srl.base.rl.registration import register

from .rainbow import Config

register(
    Config(multisteps=3).set_tensorflow(),
    __name__ + ".rainbow:Memory",
    __name__ + ".model_tf:Parameter",
    __name__ + ".model_tf:Trainer",
    __name__ + ".rainbow:Worker",
)

register(
    Config(multisteps=3).set_torch(),
    __name__ + ".rainbow:Memory",
    __name__ + ".model_torch:Parameter",
    __name__ + ".model_torch:Trainer",
    __name__ + ".rainbow:Worker",
)

register(
    Config(multisteps=1).set_tensorflow(),
    __name__ + ".rainbow:Memory",
    __name__ + ".model_tf:Parameter",
    __name__ + ".model_tf:Trainer",
    __name__ + ".rainbow_nomultisteps:Worker",
)

register(
    Config(multisteps=1).set_torch(),
    __name__ + ".rainbow:Memory",
    __name__ + ".model_torch:Parameter",
    __name__ + ".model_torch:Trainer",
    __name__ + ".rainbow_nomultisteps:Worker",
)

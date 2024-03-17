from srl.base.rl.registration import register

from .agent57_light import Config

register(
    Config().set_tensorflow(),
    __name__ + ".agent57_light:Memory",
    __name__ + ".model_tf:Parameter",
    __name__ + ".model_tf:Trainer",
    __name__ + ".agent57_light:Worker",
)

register(
    Config().set_torch(),
    __name__ + ".agent57_light:Memory",
    __name__ + ".model_torch:Parameter",
    __name__ + ".model_torch:Trainer",
    __name__ + ".agent57_light:Worker",
)

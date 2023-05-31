from srl.base.rl.registration import register

from .agent57_light import Config

register(
    Config(framework="tensorflow"),
    __name__ + ".agent57_light:RemoteMemory",
    __name__ + ".model_tf:Parameter",
    __name__ + ".model_tf:Trainer",
    __name__ + ".agent57_light:Worker",
)

register(
    Config(framework="torch"),
    __name__ + ".agent57_light:RemoteMemory",
    __name__ + ".model_torch:Parameter",
    __name__ + ".model_torch:Trainer",
    __name__ + ".agent57_light:Worker",
)

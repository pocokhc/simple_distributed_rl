from srl.base.rl.registration import register

from .rainbow import Config

register(
    Config(framework="tensorflow", multisteps=3),
    __name__ + ".rainbow:RemoteMemory",
    __name__ + ".model_tf:Parameter",
    __name__ + ".model_tf:Trainer",
    __name__ + ".rainbow:Worker",
)

register(
    Config(framework="torch", multisteps=3),
    __name__ + ".rainbow:RemoteMemory",
    __name__ + ".model_torch:Parameter",
    __name__ + ".model_torch:Trainer",
    __name__ + ".rainbow:Worker",
)

register(
    Config(framework="tensorflow", multisteps=1),
    __name__ + ".rainbow:RemoteMemory",
    __name__ + ".model_tf:Parameter",
    __name__ + ".model_tf:Trainer",
    __name__ + ".rainbow_nomultisteps:Worker",
)

register(
    Config(framework="torch", multisteps=1),
    __name__ + ".rainbow:RemoteMemory",
    __name__ + ".model_torch:Parameter",
    __name__ + ".model_torch:Trainer",
    __name__ + ".rainbow_nomultisteps:Worker",
)

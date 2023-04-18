from srl.base.rl.registration import register

from .dqn import Config

register(
    Config(framework="tensorflow"),
    __name__ + ".dqn:RemoteMemory",
    __name__ + ".model_tf:Parameter",
    __name__ + ".model_tf:Trainer",
    __name__ + ".dqn:Worker",
)

register(
    Config(framework="torch"),
    __name__ + ".dqn:RemoteMemory",
    __name__ + ".model_torch:Parameter",
    __name__ + ".model_torch:Trainer",
    __name__ + ".dqn:Worker",
)

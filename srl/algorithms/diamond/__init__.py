from srl.base.rl.registration import register

from .config import Config

register(
    Config(),
    __name__ + ".memory:Memory",
    __name__ + ".parameter:Parameter",
    __name__ + ".trainer:Trainer",
    __name__ + ".worker:Worker",
)

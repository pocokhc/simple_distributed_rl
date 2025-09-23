from srl.base.rl.registration import register as _register

from .config import Config

_register(
    Config(),
    __name__ + ".torch_model:Memory",
    __name__ + ".torch_model:Parameter",
    __name__ + ".torch_model:Trainer",
    __name__ + ".torch_model:Worker",
)

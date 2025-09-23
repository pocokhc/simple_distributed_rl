from srl.base.rl.registration import register as _register
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.np_array import NpArraySpace

from .config import Config

_register(
    Config().set_action_space(DiscreteSpace(1)),
    __name__ + ".config:Memory",
    __name__ + ".sac_tf_discrete:Parameter",
    __name__ + ".sac_tf_discrete:Trainer",
    __name__ + ".sac_tf_discrete:Worker",
)

_register(
    Config().set_action_space(NpArraySpace(1)),
    __name__ + ".config:Memory",
    __name__ + ".sac_tf_continuous:Parameter",
    __name__ + ".sac_tf_continuous:Trainer",
    __name__ + ".sac_tf_continuous:Worker",
)

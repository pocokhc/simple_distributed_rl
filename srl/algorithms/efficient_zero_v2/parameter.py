from typing import Any

import numpy as np

from srl.base.rl.parameter import RLParameter

from .config import Config
from .model import DynamicsNetwork, PredictionNetwork, ProjectorNetwork, RepresentationNetwork


class Parameter(RLParameter[Config]):
    def setup(self) -> None:
        self.representation_net = RepresentationNetwork(self.config.observation_space.shape, self.config)
        self.dynamics_net = DynamicsNetwork(self.representation_net.s_state_shape, self.config)
        self.prediction_net = PredictionNetwork(self.representation_net.s_state_shape, self.config)
        self.projector_net = ProjectorNetwork(self.representation_net.s_state_shape, self.config)
        self.q_min = np.inf
        self.q_max = -np.inf

    def call_restore(self, data: Any, **kwargs) -> None:
        self.representation_net.set_weights(data[0])
        self.dynamics_net.set_weights(data[1])
        self.prediction_net.set_weights(data[2])
        self.projector_net.set_weights(data[3])
        self.q_min = data[4]
        self.q_max = data[5]

    def call_backup(self, **kwargs):
        return [
            self.representation_net.get_weights(),
            self.dynamics_net.get_weights(),
            self.prediction_net.get_weights(),
            self.projector_net.get_weights(),
            self.q_min,
            self.q_max,
        ]

    def summary(self, **kwargs):
        self.representation_net.summary(**kwargs)
        self.dynamics_net.summary(**kwargs)
        self.prediction_net.summary(**kwargs)
        self.projector_net.summary(**kwargs)

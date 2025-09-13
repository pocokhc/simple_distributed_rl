import numpy as np

from srl.rl.memories.priority_replay_buffer import RLPriorityReplayBuffer

from .config import Config


class Memory(RLPriorityReplayBuffer[Config]):
    def setup(self) -> None:
        super().setup(register_sample=False)

        self.np_dtype = self.config.get_dtype("np")
        self.q_min = float("inf")
        self.q_max = float("-inf")
        self.register_worker_func_custom(self.add_q, lambda x1, x2: (x1, x2))
        self.register_trainer_recv_func(self.get_q)

        self.register_trainer_send_func(self.update_parameter)
        self.register_trainer_recv_func(self.sample_batch)
        self.register_trainer_send_func(self.update)

        if self.config.enable_reanalyze:
            from .mcts import MCTS
            from .parameter import Parameter

            self.parameter = Parameter(self.config)
            self.mcts = MCTS(self.config, self.parameter)

    def add_q(self, q_min, q_max, serialized: bool = False):
        self.q_min = min(self.q_min, q_min)
        self.q_max = max(self.q_max, q_max)

    def get_q(self):
        return self.q_min, self.q_max

    def sample_batch(self):
        batches = self.sample()
        if batches is None:
            return None

        batches, weights, update_args = batches

        # (batch, steps, val) -> (steps, batch, val)
        states_list = []
        actions_list = []
        policies_list = []
        z_list = []
        value_prefix_list = []
        action_dtype = np.int64 if self.config.action_space.is_discrete() else self.np_dtype
        for i in range(self.config.unroll_steps + 1):
            states_list.append(np.asarray([b[i][0] for b in batches], dtype=self.np_dtype))
            actions_list.append(np.asarray([b[i][1] for b in batches], dtype=action_dtype))
            policies_list.append(np.asarray([b[i][2] for b in batches], dtype=self.np_dtype))
            z_list.append(np.asarray([b[i][3] for b in batches], dtype=self.np_dtype))
            value_prefix_list.append(np.asarray([b[i][4] for b in batches], dtype=self.np_dtype))

        if not self.config.enable_reanalyze:
            return states_list, actions_list, value_prefix_list, policies_list, z_list, weights, update_args

        # --- reanalyze
        # TODO: 実装するならここ

        return states_list, actions_list, value_prefix_list, policies_list, z_list, weights, update_args

    def update_parameter(self, dat):
        self.parameter.restore(dat)

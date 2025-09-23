import random
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from srl.base.exception import UndefinedError
from srl.base.rl.algorithms.base_ppo import RLWorker
from srl.base.rl.parameter import RLParameter
from srl.base.rl.trainer import RLTrainer
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.np_array import NpArraySpace
from srl.rl.memories.replay_buffer import RLReplayBuffer
from srl.rl.torch_.distributions.categorical_gumbel_dist_block import CategoricalGumbelDistBlock
from srl.rl.torch_.distributions.normal_dist_block import NormalDistBlock
from srl.rl.torch_.helper import model_backup, model_restore

from .config import Config


class Memory(RLReplayBuffer):
    pass


class PolicyNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.in_block = config.input_block.create_torch_block(config)

        self.policy_block = config.policy_block.create_torch_block(self.in_block.out_size)
        if isinstance(config.action_space, DiscreteSpace):
            self.is_discrete = True
            self.policy_dist_block = CategoricalGumbelDistBlock(
                self.policy_block.out_size,
                config.action_space.n,
            )
        elif isinstance(config.action_space, NpArraySpace):
            self.is_discrete = False
            self.policy_dist_block = NormalDistBlock(
                self.policy_block.out_size,
                config.action_space.size,
                enable_stable_gradients=self.config.enable_stable_gradients,
                stable_gradients_scale_range=self.config.stable_gradients_scale_range,
            )
        else:
            raise UndefinedError(self.config.action_space)

    def forward(self, x: torch.Tensor):
        x = self.in_block(x)
        x = self.policy_block(x)
        x = self.policy_dist_block(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.in_block = config.input_block.create_torch_block(config)
        if isinstance(config.action_space, DiscreteSpace):
            self.act_block = nn.Sequential(
                nn.Linear(config.action_space.n, config.act_emb_units),
                nn.SiLU(),
            )
        else:
            self.act_block = nn.Sequential(
                nn.Linear(config.action_space.size, config.act_emb_units),
                nn.SiLU(),
            )

        self.q_block = config.q_block.create_torch_block(self.in_block.out_size + config.act_emb_units)
        self.q_out_layer = nn.Linear(self.q_block.out_size, 1)

    def forward(self, x: torch.Tensor, action: torch.Tensor):
        x = self.in_block(x)
        a = self.act_block(action)
        x = torch.cat([x, a], dim=-1)
        x = self.q_block(x)
        x = self.q_out_layer(x)
        return x


class Parameter(RLParameter[Config]):
    def setup(self):
        self.device = torch.device(self.config.used_device_torch)
        self.policy = PolicyNetwork(self.config).to(self.device)
        self.qnet = QNetwork(self.config).to(self.device)

    def call_restore(self, dat, from_serialized: bool = False, **kwargs) -> None:
        model_restore(self.policy, dat[0], from_serialized)
        model_restore(self.qnet, dat[1], from_serialized)

    def call_backup(self, serialized: bool = False, **kwargs):
        return [
            model_backup(self.policy, serialized),
            model_backup(self.qnet, serialized),
        ]

    def summary(self, **kwargs):
        print(self.policy)
        print(self.qnet)


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        self.opt_q = optim.Adam(self.parameter.qnet.parameters(), lr=self.config.lr)
        self.opt_q = self.config.lr_scheduler.apply_torch_scheduler(self.opt_q)

        self.opt_policy = optim.Adam(self.parameter.policy.parameters(), lr=self.config.lr)
        self.opt_policy = self.config.lr_scheduler.apply_torch_scheduler(self.opt_policy)

        self.np_dtype = self.config.get_dtype("np")
        self.torch_dtype = self.config.get_dtype("torch")
        self.device = self.parameter.device

        self.huber_loss = nn.HuberLoss()
        self.mse_loss = nn.MSELoss()

        act_size = self.config.action_space.n if self.config.action_space.is_discrete() else self.config.action_space.size  # type: ignore
        self.state_np = np.empty((self.config.batch_size, *self.config.observation_space.shape), dtype=self.np_dtype)
        self.n_state_np = np.empty((self.config.batch_size, *self.config.observation_space.shape), dtype=self.np_dtype)
        self.action_np = np.empty((self.config.batch_size, act_size), dtype=self.np_dtype)
        self.reward_np = np.empty((self.config.batch_size, 1), dtype=self.np_dtype)
        self.not_terminated_np = np.empty((self.config.batch_size, 1), dtype=self.np_dtype)
        self.total_reward_np = np.empty((self.config.batch_size, 1), dtype=self.np_dtype)

    def train(self) -> None:
        batches = self.memory.sample()
        if batches is None:
            return

        for i, b in enumerate(batches):
            self.state_np[i] = b[0]
            self.n_state_np[i] = b[1]
            self.action_np[i] = b[2]
            self.reward_np[i] = b[3]
            self.not_terminated_np[i] = b[4]
            self.total_reward_np[i] = b[5]
        state = torch.from_numpy(self.state_np).to(self.device)
        n_state = torch.from_numpy(self.n_state_np).to(self.device)
        action = torch.from_numpy(self.action_np).to(self.device)
        reward = torch.from_numpy(self.reward_np).to(self.device)
        not_terminated = torch.from_numpy(self.not_terminated_np).to(self.device)
        total_reward = torch.from_numpy(self.total_reward_np).to(self.device)

        with torch.no_grad():
            n_p_dist = self.parameter.policy(n_state)
            if self.config.action_space.is_discrete():
                n_action = n_p_dist.sample(temperature=self.config.target_policy_temperature, onehot=True)
                n_action = n_action.type(dtype=self.torch_dtype)
            else:
                n_action = n_p_dist.sample()
                noise = torch.normal(0, self.config.target_policy_noise_stddev, size=n_action.shape).to(self.device)
                noise = torch.clamp(noise, -self.config.target_policy_clip_range, self.config.target_policy_clip_range)
                n_action = torch.clamp(n_action + noise, -1, 1)
                if self.config.squashed_gaussian_policy:
                    n_action = torch.tanh(n_action)
            n_q = self.parameter.qnet(n_state, n_action)
            target_q = reward + not_terminated * self.config.discount * n_q

        # --- q loss
        q = self.parameter.qnet(state, action)
        loss_q = self.huber_loss(target_q, q)
        loss_q_align = self.mse_loss(total_reward, q)

        loss = loss_q + self.config.loss_align_coeff * loss_q_align
        self.info["loss_q"] = loss_q.item()
        self.info["loss_q_align"] = loss_q_align.item()

        self.opt_q.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameter.qnet.parameters(), max_norm=self.config.max_grad_norm)
        self.opt_q.step()

        # --- policy
        p_dist = self.parameter.policy(state)
        action = p_dist.rsample()
        if self.config.action_space.is_continuous() and self.config.squashed_gaussian_policy:
            action = torch.tanh(action)
        q = self.parameter.qnet(state, action)
        loss_policy = -q.mean()
        self.info["loss_policy"] = loss_policy.item()

        self.opt_policy.zero_grad()
        loss_policy.backward()
        torch.nn.utils.clip_grad_norm_(self.parameter.policy.parameters(), max_norm=self.config.max_grad_norm)
        self.opt_policy.step()

        self.train_count += 1


class Worker(RLWorker[Config, Parameter, Memory]):
    def on_setup(self, worker, context):
        self.dtype = self.config.get_dtype("np")
        self.device = self.parameter.device

    def on_reset(self, worker):
        pass

    def policy(self, worker):
        state = torch.from_numpy(worker.state[np.newaxis, ...].astype(self.dtype)).to(self.device)
        p_dist = self.parameter.policy(state)

        epsilon = self.config.epsilon if self.training else self.config.test_epsilon
        if self.config.action_space.is_discrete():
            act_space = cast(DiscreteSpace, self.config.action_space)
            if random.random() < epsilon:
                env_action = self.sample_action()
                self.action = worker.get_onehot_action(env_action)
            else:
                self.action = p_dist.sample(onehot=True)
                self.action = self.action.detach().cpu().numpy()[0]
                env_action = int(np.argmax(self.action))
        else:
            act_space = cast(NpArraySpace, self.config.action_space)
            if self.training:
                loc = p_dist.mean().detach().cpu().numpy()[0]
                scale = self.config.policy_noise_normal_scale
                self.action = np.random.normal(loc, scale, size=act_space.size)
            else:
                self.action = p_dist.mean()
                self.action = self.action.detach().cpu().numpy()[0]

            if self.config.squashed_gaussian_policy:
                self.action = np.tanh(self.action)
                env_action = act_space.rescale_from(self.action, src_low=-1, src_high=1)
            else:
                env_action = act_space.rescale_from(self.action)
                env_action = act_space.sanitize(env_action)
        return env_action

    def on_step(self, worker):
        if not self.training:
            return

        worker.add_tracking(
            {
                "state": worker.state,
                "next_state": worker.next_state,
                "action": self.action,
                "reward": worker.reward,
                "not_done": int(not worker.terminated),
            }
        )
        if worker.done:
            total_reward = 0
            for b in reversed(worker.get_trackings()):
                total_reward = b[3] + self.config.discount * total_reward
                self.memory.add(b + [total_reward])

    def render_terminal(self, worker, **kwargs) -> None:
        state = torch.from_numpy(worker.state[np.newaxis, ...].astype(self.dtype)).to(self.device)
        p_dist = self.parameter.policy(state)

        if self.config.action_space.is_discrete():
            probs = p_dist.probs().detach().cpu().numpy()[0]
            logits = p_dist.logits.detach().cpu().numpy()[0]

            def _render_sub(a: int) -> str:
                s = "{:5.1f}%".format(probs[a] * 100)
                s += f", logits {logits[a]:7.3f}"
                # --- q
                action = torch.from_numpy(np.array([worker.get_onehot_action(a)]).astype(self.dtype)).to(self.device)
                q = self.parameter.qnet(state, action)
                s += f", q {q.detach().cpu().numpy()[0][0]:.5f}"
                return s

            worker.print_discrete_action_info(int(np.argmax(probs)), _render_sub)
        else:
            action = p_dist.mean()
            q = self.parameter.qnet(state, action)
            for a in range(self.config.action_space.size):  # type: ignore
                s = f"[{a}] mean: {action.detach().cpu().numpy()[0][a]:.7f}"
                s += f", stddev: {p_dist.stddev().detach().cpu().numpy()[0][a]:.7f}"
                print(s)
            print(f"q: {q.detach().cpu().numpy()[0][0]:.7f}")

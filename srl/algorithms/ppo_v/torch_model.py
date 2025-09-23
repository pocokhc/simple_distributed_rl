import math
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
from srl.rl.functions import compute_normal_logprob, compute_normal_logprob_sgp
from srl.rl.memories.replay_buffer import RLReplayBuffer
from srl.rl.torch_.distributions.categorical_dist_block import CategoricalDistBlock
from srl.rl.torch_.distributions.normal_dist_block import NormalDistBlock
from srl.rl.torch_.helper import model_backup, model_restore

from .config import Config


class Memory(RLReplayBuffer):
    pass


class ActorCriticNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # --- input
        self.in_block = config.input_block.create_torch_block(config)

        # --- hidden block
        self.hidden_block = config.hidden_block.create_torch_block(self.in_block.out_size)

        # --- value
        self.value_block = config.value_block.create_torch_block(self.hidden_block.out_size)
        self.value_out_layer = nn.Linear(self.value_block.out_size, 1)

        # --- policy
        self.policy_block = config.policy_block.create_torch_block(self.hidden_block.out_size)
        if isinstance(config.action_space, DiscreteSpace):
            self.policy_dist_block = CategoricalDistBlock(
                self.policy_block.out_size,
                config.action_space.n,
            )
        elif isinstance(config.action_space, NpArraySpace):
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
        x = self.hidden_block(x)

        # value
        v = self.value_block(x)
        v = self.value_out_layer(v)

        # policy
        p = self.policy_block(x)
        p = self.policy_dist_block(p)
        return v, p


class Parameter(RLParameter[Config]):
    def setup(self):
        self.device = torch.device(self.config.used_device_torch)
        self.net = ActorCriticNetwork(self.config).to(self.device)

    def call_restore(self, dat, from_serialized: bool = False, **kwargs) -> None:
        model_restore(self.net, dat, from_serialized)

    def call_backup(self, serialized: bool = False, **kwargs):
        return model_backup(self.net, serialized)

    def summary(self, **kwargs):
        print(self.net)


class Trainer(RLTrainer[Config, Parameter, Memory]):
    def on_setup(self) -> None:
        self.np_dtype = self.config.get_dtype("np")
        self.device = self.parameter.device

        self.opt = optim.Adam(self.parameter.net.parameters(), lr=self.config.lr)
        self.opt = self.config.lr_scheduler.apply_torch_scheduler(self.opt)

        self.huber_loss = nn.HuberLoss()
        self.mse_loss = nn.MSELoss()

        act_size = self.config.action_space.n if self.config.action_space.is_discrete() else self.config.action_space.size  # type: ignore
        logpi_size = 1 if self.config.action_space.is_discrete() else self.config.action_space.size  # type: ignore
        self.state_np = np.empty((self.config.batch_size, *self.config.observation_space.shape), dtype=self.np_dtype)
        self.n_state_np = np.empty((self.config.batch_size, *self.config.observation_space.shape), dtype=self.np_dtype)
        self.action_np = np.empty((self.config.batch_size, act_size), dtype=self.np_dtype)
        self.old_logpi_np = np.empty((self.config.batch_size, logpi_size), dtype=self.np_dtype)
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
            self.old_logpi_np[i] = b[3]
            self.reward_np[i] = b[4]
            self.not_terminated_np[i] = b[5]
            self.total_reward_np[i] = b[6]
        state = torch.from_numpy(self.state_np).to(self.device)
        n_state = torch.from_numpy(self.n_state_np).to(self.device)
        action = torch.from_numpy(self.action_np).to(self.device)
        old_logpi = torch.from_numpy(self.old_logpi_np).to(self.device)
        reward = torch.from_numpy(self.reward_np).to(self.device)
        not_terminated = torch.from_numpy(self.not_terminated_np).to(self.device)
        total_reward = torch.from_numpy(self.total_reward_np).to(self.device)

        v, p_dist = self.parameter.net(state)
        with torch.no_grad():
            n_v, _ = self.parameter.net(n_state)

        if self.config.action_space.is_discrete():
            # onehot
            new_logpi = p_dist.log_prob(action, keepdims=True)
        else:
            if self.config.squashed_gaussian_policy:
                new_logpi = p_dist.log_prob_sgp(action)
            else:
                new_logpi = p_dist.log_prob(action)
        ratio = torch.exp(new_logpi - old_logpi)

        loss = 0

        # --- advantage
        q = reward + not_terminated * self.config.discount * n_v
        adv = (q - v).detach()

        # --- value loss
        ratio_detach = ratio.detach()
        loss_value = self.huber_loss(ratio_detach * q, v)
        loss_v_align = self.mse_loss(ratio_detach * total_reward, v)
        loss += loss_value + self.config.loss_align_coeff * loss_v_align
        self.info["loss_value"] = loss_value.item()
        self.info["loss_v_align"] = loss_v_align.item()

        # --- cliepped policy loss
        ratio_clipped = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range)
        loss_policy = torch.minimum(ratio * adv, ratio_clipped * adv)
        loss_policy = -torch.mean(loss_policy)
        loss += loss_policy
        self.info["loss_policy"] = loss_policy.item()

        if self.config.entropy_weight > 0:
            loss_e = torch.sum(-torch.exp(new_logpi) * new_logpi, dim=-1)
            loss_e = -loss_e.mean()
            loss += self.config.entropy_weight * loss_e
            self.info["loss_e"] = loss_e.item()

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameter.net.parameters(), max_norm=self.config.max_grad_norm)
        self.opt.step()

        self.train_count += 1


class Worker(RLWorker[Config, Parameter, Memory]):
    def on_setup(self, worker, context):
        self.dtype = self.config.get_dtype("np")
        self.device = self.parameter.device

    def on_reset(self, worker):
        pass

    def policy(self, worker):
        state = torch.from_numpy(worker.state[np.newaxis, ...].astype(self.dtype)).to(self.device)
        v, p_dist = self.parameter.net(state)

        epsilon = self.config.epsilon if self.training else self.config.test_epsilon
        if self.config.action_space.is_discrete():
            act_space = cast(DiscreteSpace, self.config.action_space)
            if random.random() < epsilon:
                env_action = self.sample_action()
                self.action = worker.get_onehot_action(env_action)
                self.log_prob = [math.log(1.0 / act_space.n)]
            else:
                self.action = p_dist.sample(onehot=True)
                self.log_prob = p_dist.log_prob(self.action)
                self.log_prob = self.log_prob.detach().cpu().numpy()[0]
                self.action = self.action.detach().cpu().numpy()[0]
                env_action = int(np.argmax(self.action))
        elif self.config.action_space.is_continuous():
            act_space = cast(NpArraySpace, self.config.action_space)
            loc = p_dist.mean().detach().cpu().numpy()[0]
            scale = p_dist.stddev().detach().cpu().numpy()[0]
            if self.training:
                if random.random() < epsilon:
                    loc = 0
                    scale = self.config.policy_noise_normal_scale
                self.action = np.random.normal(loc, scale, size=act_space.size)
            else:
                self.action = p_dist.mean()
                self.action = self.action.detach().cpu().numpy()[0]

            if self.config.squashed_gaussian_policy:
                self.log_prob = compute_normal_logprob_sgp(self.action, loc, scale)
                env_action = np.tanh(self.action)
                env_action = act_space.rescale_from(env_action, src_low=-1, src_high=1)
            else:
                self.log_prob = compute_normal_logprob(self.action, loc, scale)
                env_action = act_space.rescale_from(self.action)
                env_action = act_space.sanitize(env_action)
        else:
            raise UndefinedError(self.config.action_space)
        return env_action

    def on_step(self, worker):
        if not self.training:
            return

        worker.add_tracking(
            {
                "state": worker.state,
                "next_state": worker.next_state,
                "action": self.action,
                "log_prob": np.maximum(self.log_prob, math.log(1e-6)),  # 0除算回避用
                "reward": worker.reward,
                "not_done": int(not worker.terminated),
            }
        )
        if worker.done:
            total_reward = 0
            for b in reversed(worker.get_trackings()):
                total_reward = b[4] + self.config.discount * total_reward
                self.memory.add(b + [total_reward])

    def render_terminal(self, worker, **kwargs) -> None:
        state = torch.from_numpy(worker.state[np.newaxis, ...].astype(self.dtype)).to(self.device)
        v, p_dist = self.parameter.net(state)
        print(f"V: {v.detach().cpu().numpy()[0][0]:.7f}")

        if self.config.action_space.is_discrete():
            probs = p_dist.probs().detach().cpu().numpy()[0]
            logits = p_dist.logits().detach().cpu().numpy()[0]

            def _render_sub(a: int) -> str:
                s = "{:5.1f}%".format(probs[a] * 100)
                s += f", logits {logits[a]:.7f}"
                return s

            worker.print_discrete_action_info(int(np.argmax(probs)), _render_sub)
        elif self.config.action_space.is_continuous():
            for a in range(self.config.action_space.size):  # type: ignore
                s = f"[{a}] mean: {p_dist.mean().detach().cpu().numpy()[0][a]:.7f}"
                s += f", stddev: {p_dist.stddev().detach().cpu().numpy()[0][a]:.7f}"
                print(s)
        else:
            raise UndefinedError(self.config.action_space)

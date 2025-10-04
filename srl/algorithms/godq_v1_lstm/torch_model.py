import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from srl.algorithms.godq_v1.torch_model_nets import BYOLNetwork, EmbeddingNetwork, QNetwork, RNDModel
from srl.rl.torch_.helper import model_backup, model_restore

from .config import Config

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    def __init__(self, config: Config, device):
        super().__init__()
        base_units = config.base_units
        self.torch_dtype = config.get_dtype("torch")
        self.lstm_c_clip = config.lstm_c_clip
        self.device = device

        self.obs_encoder = config.input_block.create_torch_block(config.observation_space)
        self.act_encoder = nn.Embedding(config.action_space.n, base_units)

        self.lstm_units = base_units
        self.lstm = nn.LSTMCell(self.obs_encoder.out_size + base_units, self.lstm_units)
        self.out_size: int = self.lstm_units

    def forward(self, state, action_indices: torch.Tensor, hc):
        ts = self.obs_encoder(state)
        ta = self.act_encoder(action_indices)
        z = torch.cat([ts, ta], dim=-1)
        h, c = self.lstm(z, hc)
        if self.lstm_c_clip > 0:
            c = torch.clamp(c, -self.lstm_c_clip, self.lstm_c_clip)
        return h, (h, c)

    def forward_encode(self, state, action_indices: torch.Tensor):
        ts = self.obs_encoder(state)
        ta = self.act_encoder(action_indices)
        return torch.cat([ts, ta], dim=-1)

    def forward_lstm(self, z: torch.Tensor, hc):
        h, c = self.lstm(z, hc)
        if self.lstm_c_clip > 0:
            c = torch.clamp(c, -self.lstm_c_clip, self.lstm_c_clip)
        return h, (h, c)

    def get_initial_state(self, batch_size=1):
        return (
            torch.zeros(batch_size, self.lstm_units, dtype=self.torch_dtype, device=self.device),
            torch.zeros(batch_size, self.lstm_units, dtype=self.torch_dtype, device=self.device),
        )


class Model:
    def __init__(self, config: Config):
        self.config = config
        self.torch_dtype = config.get_dtype("torch")
        self.np_dtype = config.get_dtype("np")
        self.device = torch.device(config.used_device_torch)

        # --- Q
        self.encoder = Encoder(config, self.device).to(self.device)
        enc_out_size = self.encoder.out_size
        self.q_online = QNetwork(enc_out_size, config).to(self.device)

        # --- feat
        if self.config.feat_type == "BYOL":
            self.byol_online = BYOLNetwork(enc_out_size, config).to(self.device)
            self.byol_target = self.byol_online.create_projection().to(self.device)
            self.byol_target.load_state_dict(self.byol_online.proj_block.state_dict())
            self.byol_target.eval()
            for p in self.byol_target.parameters():
                p.requires_grad_(False)

        # --- q int
        if self.config.enable_int_q:
            self.q_int_online = QNetwork(enc_out_size, config).to(self.device)
            if self.config.feat_type == "":
                self.rnd = RNDModel(enc_out_size, config).to(self.device)
            if self.config.enable_int_episodic:
                self.emb_net = EmbeddingNetwork(enc_out_size, config.base_units, config.action_space.n).to(self.device)

    def restore(self, dat, from_serialized: bool) -> None:
        model_restore(self.encoder, dat["encoder"], from_serialized)
        model_restore(self.q_online, dat["q_online"], from_serialized)
        if self.config.feat_type == "BYOL":
            model_restore(self.byol_online, dat["byol_online"], from_serialized)
            model_restore(self.byol_target, dat["byol_target"], from_serialized)
        if self.config.enable_int_q:
            model_restore(self.q_int_online, dat["q_int_online"], from_serialized)
            if self.config.feat_type == "":
                model_restore(self.rnd, dat["rnd"], from_serialized)
            if self.config.enable_int_episodic:
                model_restore(self.emb_net, dat["emb_net"], from_serialized)

    def backup(self, serialized: bool):
        dat: dict = {
            "encoder": model_backup(self.encoder, serialized),
            "q_online": model_backup(self.q_online, serialized),
        }
        if self.config.feat_type == "BYOL":
            dat["byol_online"] = model_backup(self.byol_online, serialized)
            dat["byol_target"] = model_backup(self.byol_target, serialized)
        if self.config.enable_int_q:
            dat["q_int_online"] = model_backup(self.q_int_online, serialized)
            if self.config.feat_type == "":
                dat["rnd"] = model_backup(self.rnd, serialized)
            if self.config.enable_int_episodic:
                dat["emb_net"] = model_backup(self.emb_net, serialized)
        return dat

    def summary(self, **kwargs):
        print(self.encoder)
        print(self.q_online)
        if self.config.enable_int_q:
            if self.config.feat_type == "":
                print(self.rnd)
            print(self.q_int_online)
            if self.config.enable_int_episodic:
                print(self.emb_net)
        if self.config.feat_type == "BYOL":
            print(self.byol_online)

    # -------------------
    def pred_oe(self, states: list, action: int, hc):
        inputs = [
            torch.tensor(state[np.newaxis, ...], dtype=self.torch_dtype, device=self.device)
            for state in states  #
        ]
        ta = torch.tensor(np.array([action]), dtype=torch.long, device=self.device)
        with torch.no_grad():
            self.encoder.eval()
            z, hc = self.encoder(inputs, ta, hc)
            self.encoder.train()
        return z, hc

    def pred_q(self, oe: torch.Tensor, is_mean: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            self.q_online.eval()
            if is_mean:
                q, v = self.q_online.forward_mean(oe)
            else:
                q, v = self.q_online(oe)
            self.q_online.train()  # 常にtrain
        return q.detach().cpu().numpy(), v.detach().cpu().numpy()

    def pred_q_int(self, oe: torch.Tensor, is_mean: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            self.q_int_online.eval()
            if is_mean:
                q, v = self.q_int_online.forward_mean(oe)
            else:
                q, v = self.q_int_online(oe)
            self.q_int_online.train()  # 常にtrain
        return q.detach().cpu().numpy(), v.detach().cpu().numpy()

    def pred_single_int_reward(self, oe, n_oe, action: int) -> np.ndarray:
        with torch.no_grad():
            action_indices = torch.tensor(np.asarray([action]), dtype=torch.long, device=self.device)
            if self.config.feat_type == "":
                self.rnd.eval()
                int_rew = self.rnd.compute_intrinsic_reward(n_oe)
                self.rnd.train()
            elif self.config.feat_type == "BYOL":
                self.byol_online.eval()
                y_hat = self.byol_online(oe, action_indices)
                y_target = self.byol_target(n_oe)
                _, int_rew = self.byol_online.compute_loss_and_reward(y_target, y_hat)
                self.byol_online.train()
            int_rew = int_rew.detach().cpu().numpy()[0]
        return int_rew

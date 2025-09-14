import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from srl.algorithms.godq_v1.torch_model_encoder import create_encoder_block
from srl.algorithms.godq_v1.torch_model_feat import BYOLNetwork, ProjectorNetwork
from srl.algorithms.godq_v1.torch_model_q import QIntNetwork, QNetwork
from srl.rl.torch_.functions import inverse_linear_symlog
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

        self.obs_encoder, enc_out_size = create_encoder_block(config)
        self.act_encoder = nn.Embedding(config.action_space.n, base_units)

        self.lstm_units = base_units
        self.lstm = nn.LSTMCell(enc_out_size + base_units, self.lstm_units)
        self.out_size: int = self.lstm_units

    def forward(self, state: torch.Tensor, action_indices: torch.Tensor, hc):
        ts = self.obs_encoder(state)
        ta = self.act_encoder(action_indices)
        z = torch.cat([ts, ta], dim=-1)
        h, c = self.lstm(z, hc)
        if self.lstm_c_clip > 0:
            c = torch.clamp(c, -self.lstm_c_clip, self.lstm_c_clip)
        return h, (h, c)

    def forward_encode(self, state: torch.Tensor, action_indices: torch.Tensor):
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
        self.q_online = QNetwork(self.encoder.out_size, config, config.enable_q_distribution).to(self.device)

        if self.config.feat_type == "SimSiam":
            self.projector = ProjectorNetwork(config.base_units, self.encoder.out_size, config.action_space.n).to(self.device)
        elif self.config.feat_type == "BYOL":
            self.byol_online = BYOLNetwork(config.base_units, self.encoder.out_size, config.action_space.n).to(self.device)
            self.byol_target = self.byol_online.create_projection().to(self.device)
            self.byol_target.load_state_dict(self.byol_online.proj_block.state_dict())
            self.byol_target.eval()
            for p in self.byol_target.parameters():
                p.requires_grad_(False)

        if self.config.enable_int_q:
            assert self.config.feat_type != ""
            self.q_int_online = QIntNetwork(self.encoder.out_size, config).to(self.device)

    def restore(self, dat, from_serialized: bool) -> None:
        model_restore(self.encoder, dat["encoder"], from_serialized)
        model_restore(self.q_online, dat["q_online"], from_serialized)
        if self.config.feat_type == "SimSiam":
            model_restore(self.projector, dat["projector"], from_serialized)
        elif self.config.feat_type == "BYOL":
            model_restore(self.byol_online, dat["byol_online"], from_serialized)
            model_restore(self.byol_target, dat["byol_target"], from_serialized)
        if self.config.enable_int_q:
            model_restore(self.q_int_online, dat["q_int_online"], from_serialized)

    def backup(self, serialized: bool):
        dat: dict = {
            "encoder": model_backup(self.encoder, serialized),
            "q_online": model_backup(self.q_online, serialized),
        }
        if self.config.feat_type == "SimSiam":
            dat["projector"] = model_backup(self.projector, serialized)
        elif self.config.feat_type == "BYOL":
            dat["byol_online"] = model_backup(self.byol_online, serialized)
            dat["byol_target"] = model_backup(self.byol_target, serialized)
        if self.config.enable_int_q:
            dat["q_int_online"] = model_backup(self.q_int_online, serialized)
        return dat

    def summary(self, **kwargs):
        print(self.encoder)
        print(self.q_online)
        if self.config.enable_int_q:
            print(self.q_int_online)
        if self.config.feat_type == "SimSiam":
            print(self.projector)
        elif self.config.feat_type == "BYOL":
            print(self.byol_online)

    # -------------------
    def pred_oe(self, state: np.ndarray, action: List[int], hc):
        ts = torch.tensor(state, dtype=self.torch_dtype, device=self.device)
        ta = torch.tensor(np.asarray(action), dtype=torch.long, device=self.device)
        with torch.no_grad():
            self.encoder.eval()
            z, hc = self.encoder(ts, ta, hc)
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
        if self.config.enable_q_rescale:
            q = inverse_linear_symlog(q)
            v = inverse_linear_symlog(v)
        return q.detach().cpu().numpy(), v.detach().cpu().numpy()

    def pred_q_int(self, oe: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            self.q_int_online.eval()
            q = self.q_int_online(oe)
            self.q_int_online.train()  # 常にtrain
        return q.detach().cpu().numpy()

    def pred_single_int_reward(self, oe, n_oe, action: int) -> np.ndarray:
        with torch.no_grad():
            action_indices = torch.tensor(np.asarray([action]), dtype=torch.long, device=self.device)
            if self.config.feat_type == "SimSiam":
                self.projector.eval()
                y_hat = self.projector(oe, action_indices)
                y_target = self.projector.projection(n_oe)
                _, int_rew = self.projector.compute_loss_and_reward(y_target, y_hat)
                self.projector.train()
            elif self.config.feat_type == "BYOL":
                self.byol_online.eval()
                y_hat = self.byol_online(oe, action_indices)
                y_target = self.byol_target(n_oe)
                _, int_rew = self.byol_online.compute_loss_and_reward(y_target, y_hat)
                self.byol_online.train()
            int_rew = int_rew.detach().cpu().numpy()[0]
        return int_rew

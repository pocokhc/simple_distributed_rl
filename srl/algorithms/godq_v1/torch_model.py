import logging

import numpy as np
import torch

from srl.base.rl.worker_run import WorkerRun
from srl.rl.torch_.helper import model_backup, model_restore, model_sync

from .config import Config
from .torch_model_feat import ProjectorNetwork, SPRNetwork
from .torch_models import AE, QNetwork, RNDNetwork

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, config: Config):
        self.config = config
        self.torch_dtype = config.get_dtype("torch")
        self.np_dtype = config.get_dtype("np")
        self.device = torch.device(config.used_device_torch)

        # --- Q
        self.q_online = QNetwork(config, config.base_units, config.action_space.n, self.device).to(self.device)
        self.q_target = QNetwork(config, config.base_units, config.action_space.n, self.device).to(self.device)
        self.q_target.eval()
        if self.config.init_target_q_zero:
            for param in self.q_target.parameters():
                param.data.copy_(param.data * 0)
        else:
            self.q_target.load_state_dict(self.q_online.state_dict())
        enc_out_size = self.q_online.enc_out_size

        # --- rnd
        if config.discount <= 0:
            self.rnd_train = RNDNetwork(enc_out_size, config.base_units).to(self.device)
            self.rnd_target = RNDNetwork(enc_out_size, config.base_units).to(self.device)
            self.rnd_target.eval()
            self.rnd_max = 0.0

        # --- latent
        if config.enable_archive:
            lat_units = min(int(config.base_units // 2), 128)
            self.latent_encoder = AE(enc_out_size, lat_units, config.latent_size).to(self.device)

        # --- feat
        if self.config.feat_type == "SimSiam":
            self.projector = ProjectorNetwork(config.base_units, enc_out_size, config.action_space.n).to(self.device)
        elif self.config.feat_type == "SPR":
            self.spr = SPRNetwork(config.base_units, enc_out_size, config.action_space.n).to(self.device)
            self.project_target_block = self.spr.create_projection().to(self.device)

    def restore(self, dat, from_serialized: bool) -> None:
        model_restore(self.q_online, dat["q_online"], from_serialized)
        model_restore(self.q_target, dat["q_target"], from_serialized)
        if self.config.discount <= 0:
            model_restore(self.rnd_train, dat["rnd_train"], from_serialized)
            model_restore(self.rnd_target, dat["rnd_target"], from_serialized)
            self.rnd_max = dat["rnd_max"]
        if self.config.enable_archive:
            model_restore(self.latent_encoder, dat["latent_encoder"], from_serialized)
        if self.config.feat_type == "SimSiam":
            model_restore(self.projector, dat["projector"], from_serialized)
        elif self.config.feat_type == "SPR":
            model_restore(self.spr, dat["spr"], from_serialized)
            model_sync(self.project_target_block, self.spr.project_block)

    def backup(self, serialized: bool):
        dat: dict = {
            "q_online": model_backup(self.q_online, serialized),
            "q_target": model_backup(self.q_target, serialized),
        }
        if self.config.discount <= 0:
            dat["rnd_train"] = model_backup(self.rnd_train, serialized)
            dat["rnd_target"] = model_backup(self.rnd_target, serialized)
            dat["rnd_max"] = self.rnd_max
        if self.config.enable_archive:
            dat["latent_encoder"] = model_backup(self.latent_encoder, serialized)
        if self.config.feat_type == "SimSiam":
            dat["projector"] = model_backup(self.projector, serialized)
        elif self.config.feat_type == "SPR":
            dat["spr"] = model_backup(self.spr, serialized)
        return dat

    def summary(self, **kwargs):
        print(self.q_online)
        if self.config.discount <= 0:
            print(self.rnd_train)
        if self.config.enable_archive:
            print(self.latent_encoder)
        if self.config.feat_type == "SimSiam":
            print(self.projector)
        elif self.config.feat_type == "SPR":
            print(self.spr)

    # -------------------
    def encode_obs(self, state: np.ndarray, select_model: str = ""):
        if select_model == "":
            select_model = self.config.select_model

        z = torch.tensor(state, dtype=self.torch_dtype, device=self.device)
        with torch.no_grad():
            if select_model == "target":
                oe = self.q_target.encoder(z)
            else:
                self.q_online.eval()
                oe = self.q_online.encoder(z)
                self.q_online.train()  # 常にtrain
        return oe

    def pred_q(self, state: np.ndarray, select_model: str = ""):
        if select_model == "":
            select_model = self.config.select_model

        z = torch.tensor(state, dtype=self.torch_dtype, device=self.device)
        with torch.no_grad():
            if select_model == "target":
                oe, q = self.q_target(z)
            else:
                self.q_online.eval()
                oe, q = self.q_online(z)
                self.q_online.train()  # 常にtrain
            q = q.detach().cpu().numpy()
        return oe, q

    def encode_latent(self, oe: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            self.latent_encoder.eval()
            z = self.latent_encoder.encode(oe)
            self.latent_encoder.train()  # 常にtrain
            oz = z.detach().cpu().numpy()
        return oz

    # -------------------

    def render_terminal(self, worker: WorkerRun):
        # --- q
        oe_online, q_online = self.pred_q(worker.state[np.newaxis, ...], "online")
        q_online = q_online[0]
        oe_target, q_target = self.pred_q(worker.state[np.newaxis, ...], "target")
        q_target = q_target[0]

        if self.config.discount <= 0:
            with torch.no_grad():
                target_val = self.rnd_target(oe_online)
                train_val = self.rnd_train(oe_online)
            error = ((target_val - train_val) ** 2).mean(dim=1)
            error = error.detach().cpu().numpy()[0]
            discount = 1 - error / self.rnd_max
            print(f"discount: {discount}, rnd_max: {self.rnd_max:.5f}")

        def _render_sub(a: int) -> str:
            s = f"{q_online[a]:6.3f}(online)"
            s += f" {q_target[a]:6.3f}(target)"
            return s

        worker.print_discrete_action_info(worker.action, _render_sub)

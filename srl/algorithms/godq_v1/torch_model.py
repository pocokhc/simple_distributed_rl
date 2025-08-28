import logging

import numpy as np
import torch

from srl.base.rl.worker_run import WorkerRun
from srl.rl.torch_.helper import model_backup, model_restore

from .config import Config
from .torch_model_feat import BYOLNetwork, ProjectorNetwork
from .torch_model_q import QIntNetwork, QNetwork

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, config: Config):
        self.config = config
        self.torch_dtype = config.get_dtype("torch")
        self.np_dtype = config.get_dtype("np")
        self.device = torch.device(config.used_device_torch)

        # --- Q
        self.q_online = QNetwork(config, config.base_units, config.action_space.n, self.device).to(self.device)
        enc_out_size = self.q_online.enc_out_size

        if self.config.feat_type == "SimSiam":
            self.projector = ProjectorNetwork(config.base_units, enc_out_size, config.action_space.n).to(self.device)
        elif self.config.feat_type == "BYOL":
            self.byol_online = BYOLNetwork(config.base_units, enc_out_size, config.action_space.n).to(self.device)
            self.byol_target = self.byol_online.create_projection().to(self.device)
            self.byol_target.load_state_dict(self.byol_online.proj_block.state_dict())
            self.byol_target.eval()
            for p in self.byol_target.parameters():
                p.requires_grad_(False)

        if self.config.enable_int_q:
            assert self.config.feat_type != ""
            self.q_int_online = QIntNetwork(enc_out_size, config.base_units, config.action_space.n).to(self.device)

    def restore(self, dat, from_serialized: bool) -> None:
        model_restore(self.q_online, dat["q_online"], from_serialized)
        if self.config.feat_type == "SimSiam":
            model_restore(self.projector, dat["projector"], from_serialized)
        elif self.config.feat_type == "BYOL":
            model_restore(self.byol_online, dat["byol_online"], from_serialized)
            model_restore(self.byol_target, dat["byol_target"], from_serialized)
        if self.config.enable_int_q:
            model_restore(self.q_int_online, dat["q_int_online"], from_serialized)

    def backup(self, serialized: bool):
        dat: dict = {"q_online": model_backup(self.q_online, serialized)}
        if self.config.feat_type == "SimSiam":
            dat["projector"] = model_backup(self.projector, serialized)
        elif self.config.feat_type == "BYOL":
            dat["byol_online"] = model_backup(self.byol_online, serialized)
            dat["byol_target"] = model_backup(self.byol_target, serialized)
        if self.config.enable_int_q:
            dat["q_int_online"] = model_backup(self.q_int_online, serialized)
        return dat

    def summary(self, **kwargs):
        print(self.q_online)
        if self.config.feat_type == "SimSiam":
            print(self.projector)
        elif self.config.feat_type == "BYOL":
            print(self.byol_online)
        if self.config.enable_int_q:
            print(self.q_int_online)

    # -------------------
    def pred_oe(self, state: np.ndarray):
        z = torch.tensor(state, dtype=self.torch_dtype, device=self.device)
        with torch.no_grad():
            self.q_online.eval()
            oe = self.q_online.encoder(z)
            self.q_online.train()  # 常にtrain
        return oe

    def pred_q(self, state: np.ndarray):
        z = torch.tensor(state, dtype=self.torch_dtype, device=self.device)
        with torch.no_grad():
            self.q_online.eval()
            oe, q = self.q_online(z)
            self.q_online.train()  # 常にtrain
        q = q.detach().cpu().numpy()
        return oe, q

    def pred_q_int(self, oe: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            self.q_int_online.eval()
            q_int = self.q_int_online(oe)
            self.q_int_online.train()  # 常にtrain
        return q_int.detach().cpu().numpy()

    def pred_single_int_reward(self, state: np.ndarray, action: int, next_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            action_indices = torch.tensor(np.asarray([action]), dtype=torch.long, device=self.device)
            oe = self.pred_oe(state[np.newaxis, ...])
            n_oe = self.pred_oe(next_state[np.newaxis, ...])
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

    # -------------------

    def render_terminal(self, worker: WorkerRun):
        # --- q
        print("--- q")
        oe_online, q_online = self.pred_q(worker.state[np.newaxis, ...])
        q_online = q_online[0]

        def _render_sub(a: int) -> str:
            s = f"{q_online[a]:6.3f}(online)"
            return s

        worker.print_discrete_action_info(int(np.argmax(q_online)), _render_sub)

        # --- q int
        if self.config.enable_int_q:
            print("--- q int")
            q_int_online = self.pred_q_int(oe_online)
            q_int_online = q_int_online[0]

            def _render_sub2(a: int) -> str:
                s = f"{q_int_online[a]:6.3f}(online)"
                return s

            worker.print_discrete_action_info(int(np.argmax(q_int_online)), _render_sub2)

import logging

import numpy as np
import torch

from srl.base.rl.worker_run import WorkerRun
from srl.rl.torch_.helper import model_backup, model_restore

from .config import Config
from .torch_model_encoder import create_encoder_block
from .torch_model_feat import BYOLNetwork, ProjectorNetwork
from .torch_model_q import QNetwork

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, config: Config):
        self.config = config
        self.torch_dtype = config.get_dtype("torch")
        self.np_dtype = config.get_dtype("np")
        self.device = torch.device(config.used_device_torch)

        # --- Q
        self.encoder, enc_out_size = create_encoder_block(config)
        self.encoder.to(self.device)
        self.q_online = QNetwork(enc_out_size, config, config.enable_q_distribution, duel_net=True).to(self.device)

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
            self.q_int_online = QNetwork(enc_out_size, config, config.int_q_distribution, duel_net=False).to(self.device)

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
    def pred_oe(self, state: np.ndarray):
        z = torch.tensor(state, dtype=self.torch_dtype, device=self.device)
        with torch.no_grad():
            self.encoder.eval()
            oe = self.encoder(z)
            self.encoder.train()  # 常にtrain
        return oe

    def pred_q(self, oe: torch.Tensor, is_mean: bool = False) -> np.ndarray:
        with torch.no_grad():
            self.q_online.eval()
            if is_mean:
                q = self.q_online.forward_mean(oe)
            else:
                q = self.q_online(oe)
            self.q_online.train()  # 常にtrain
        q = q.detach().cpu().numpy()
        return q

    def pred_q_int(self, oe: torch.Tensor, is_mean: bool = False) -> np.ndarray:
        with torch.no_grad():
            self.q_int_online.eval()
            if is_mean:
                q = self.q_int_online.forward_mean(oe)
            else:
                q = self.q_int_online(oe)
            self.q_int_online.train()  # 常にtrain
        return q.detach().cpu().numpy()

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
        oe = self.pred_oe(worker.state[np.newaxis, ...])
        q = self.pred_q(oe, is_mean=True)[0]
        if self.config.enable_q_distribution:
            q_dist, v_dist, adv_dist = self.q_online.get_distribution(oe)
            v_mean = v_dist.mean().detach().cpu().numpy()[0][0]
            v_stdev = v_dist.stddev().detach().cpu().numpy()[0][0]
            adv_mean = adv_dist.mean().detach().cpu().numpy()[0]
            adv_stdev = adv_dist.stddev().detach().cpu().numpy()[0]
            q_mean = q_dist.mean().detach().cpu().numpy()[0]
            q_stdev = q_dist.stddev().detach().cpu().numpy()[0]
            print(f" V: {v_mean:.7f}(sigma: {v_stdev:.5f})")

        def _render_sub(a: int) -> str:
            s = f"{q[a]:6.3f}"
            if self.config.enable_q_distribution:
                s += f" q({q_mean[a]:6.3f}, {q_stdev[a]:6.3f})"
                s += f" adv({adv_mean[a]:6.3f}, {adv_stdev[a]:6.3f})"
            return s

        worker.print_discrete_action_info(int(np.argmax(q)), _render_sub)

        # --- q int
        if self.config.enable_int_q:
            print("--- q int")
            q_int = self.pred_q_int(oe, is_mean=True)[0]
            if self.config.int_q_distribution:
                q_dist, v_dist, adv_dist = self.q_int_online.get_distribution(oe)
                v_mean = v_dist.mean().detach().cpu().numpy()[0][0]
                v_stdev = v_dist.stddev().detach().cpu().numpy()[0][0]
                adv_mean = adv_dist.mean().detach().cpu().numpy()[0]
                adv_stdev = adv_dist.stddev().detach().cpu().numpy()[0]
                q_mean = q_dist.mean().detach().cpu().numpy()[0]
                q_stdev = q_dist.stddev().detach().cpu().numpy()[0]
                print(f" V: {v_mean:.7f}(sigma: {v_stdev:.5f})")

            def _render_sub2(a: int) -> str:
                s = f"{q_int[a]:6.3f}"
                if self.config.int_q_distribution:
                    s += f" q({q_mean[a]:6.3f}, {q_stdev[a]:6.3f})"
                    s += f" adv({adv_mean[a]:6.3f}, {adv_stdev[a]:6.3f})"
                return s

            worker.print_discrete_action_info(int(np.argmax(q_int)), _render_sub2)

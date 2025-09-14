import logging
from typing import Any

from srl.base.rl.parameter import RLParameter
from srl.rl.torch_.helper import model_backup, model_restore

from .archive import Archive
from .config import Config
from .torch_diffusion import Denoiser, DiffusionSampler
from .torch_model import Model

logger = logging.getLogger(__name__)


class Parameter(RLParameter[Config]):
    def setup(self):
        self.net = Model(self.config)
        if self.config.enable_diffusion:
            import torch

            device = torch.device(self.config.used_device_torch)
            self.denoiser = Denoiser(
                self.config.obs_render_img_space.shape,
                self.config.action_space.n,
                self.config.denoiser,
            ).to(device)
            self.sampler = DiffusionSampler(
                self.denoiser,
                self.config.sampler,
                self.config.get_dtype("torch"),
                device,
            )
        if self.config.enable_archive:
            self.archive = Archive(self.config, self.net)

    def call_restore(self, dat: dict, from_serialized: bool = False, from_worker: bool = False, **kwargs) -> None:
        self.net.restore(dat["net"], from_serialized)
        if self.config.enable_diffusion:
            model_restore(self.denoiser, dat["denoiser"], from_serialized)

        if (not from_serialized) or from_worker:
            if self.config.enable_archive:
                self.archive.restore(dat["archive"])

    def call_backup(self, serialized: bool = False, to_worker=False, **kwargs) -> Any:
        dat: dict = {"net": self.net.backup(serialized)}
        if self.config.enable_diffusion:
            dat["denoiser"] = model_backup(self.denoiser, serialized)

        # seralizedでは使わないけどto_workerなら使う
        if (not serialized) or to_worker:
            if self.config.enable_archive:
                dat["archive"] = self.archive.backup()
        return dat

    def update_from_worker_parameter(self, worker_parameger: "Parameter"):
        if self.config.enable_archive:
            self.archive.restore(worker_parameger.archive.backup())

    def summary(self, **kwargs):
        self.net.summary(**kwargs)
        if self.config.enable_diffusion:
            print(self.denoiser)

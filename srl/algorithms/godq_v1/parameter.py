import logging
from typing import Any

from srl.base.rl.parameter import RLParameter

from .archive import Archive
from .config import Config
from .torch_model import Model

logger = logging.getLogger(__name__)


class Parameter(RLParameter[Config]):
    def setup(self):
        self.net = Model(self.config)
        if self.config.enable_archive:
            self.archive = Archive(self.config, self.net)

    def call_restore(self, dat: dict, from_serialized: bool = False, from_worker: bool = False, **kwargs) -> None:
        self.net.restore(dat["net"], from_serialized)
        if (not from_serialized) or from_worker:
            if self.config.enable_archive:
                self.archive.restore(dat["archive"])

    def call_backup(self, serialized: bool = False, to_worker=False, **kwargs) -> Any:
        dat: dict = {"net": self.net.backup(serialized)}

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

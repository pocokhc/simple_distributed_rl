import logging
from abc import ABC

from srl.runner.callback import Callback

logger = logging.getLogger(__name__)


class MPCallback(Callback, ABC):
    # all
    def on_init(self, **kwargs) -> None:
        pass  # do nothing

    # main
    def on_start(self, **kwargs) -> None:
        pass  # do nothing

    def on_polling(self, **kwargs) -> None:
        pass  # do nothing

    def on_end(self, **kwargs) -> None:
        pass  # do nothing

    # trainer
    def on_trainer_start(self, **kwargs) -> None:
        pass  # do nothing

    def on_trainer_train_end(self, **kwargs) -> None:
        pass  # do nothing

    def on_trainer_end(self, **kwargs) -> None:
        pass  # do nothing

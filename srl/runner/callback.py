import logging
from abc import ABC

logger = logging.getLogger(__name__)


class Callback(ABC):
    def on_episodes_begin(self, **kwargs) -> None:
        pass  # do nothing

    def on_episodes_end(self, **kwargs) -> None:
        pass  # do nothing

    def on_episode_begin(self, **kwargs) -> None:
        pass  # do nothing

    def on_episode_end(self, **kwargs) -> None:
        pass  # do nothing

    def on_step_begin(self, **kwargs) -> None:
        pass  # do nothing

    def on_step_end(self, **kwargs) -> None:
        pass  # do nothing

    def on_skip_step(self, **kwargs) -> None:
        pass  # do nothing

    # 外部から途中停止用
    def intermediate_stop(self, **kwargs) -> bool:
        return False

import logging
import traceback
from typing import Dict, Union

from srl.base.env.base import EnvBase
from srl.base.env.config import EnvConfig
from srl.base.env.env_run import EnvRun
from srl.base.exception import UndefinedError
from srl.utils.common import is_package_installed, load_module

logger = logging.getLogger(__name__)

_registry = {}


def make_base(config: Union[str, EnvConfig]) -> EnvBase:
    if isinstance(config, str):
        config = EnvConfig(config)

    env_name = config.name

    if env_name not in _registry:
        # --- srl内のenvはloadする
        if env_name in ["Grid", "EasyGrid"]:
            import srl.envs.grid  # noqa F401
        elif env_name == "IGrid":
            import srl.envs.igrid  # noqa F401
        elif env_name == "connectx":
            import srl.envs.connectx  # noqa F401
        elif env_name in ["OneRoad", "OneRoad-hard"]:
            import srl.envs.oneroad  # noqa F401
        elif env_name in ["Othello", "Othello6x6", "Othello4x4"]:
            import srl.envs.othello  # noqa F401
        elif env_name == "OX":
            import srl.envs.ox  # noqa F401
        elif env_name == "SampleEnv":
            import srl.envs.sample_env  # noqa F401
        elif env_name == "StoneTaking":
            import srl.envs.stone_taking  # noqa F401
        elif env_name == "Tiger":
            import srl.envs.tiger  # noqa F401

    env = None

    # --- load register
    if env_name in _registry:
        env_cls = load_module(_registry[env_name]["entry_point"])

        _kwargs = _registry[env_name]["kwargs"].copy()
        _kwargs.update(config.kwargs)
        env = env_cls(**_kwargs)

    # --- load gymnasium
    if env is None and is_package_installed("gymnasium"):
        try:
            from srl.base.env.gymnasium_wrapper import GymnasiumWrapper

            env = GymnasiumWrapper(config)
        except Exception as e:
            logger.info(traceback.format_exc())
            logger.warning(f"Gymnasium failed to load. '{e}'")

    # --- load gym
    if env is None and is_package_installed("gym"):
        try:
            from srl.base.env.gym_wrapper import GymWrapper

            env = GymWrapper(config)
        except Exception as e:
            logger.info(traceback.format_exc())
            logger.warning(f"Gym failed to load. '{e}'")

    if env is None:
        raise UndefinedError(f"'{env_name}' is not found.")

    return env


def make(config: Union[str, EnvConfig]) -> EnvRun:
    if isinstance(config, str):
        config = EnvConfig(config)
    return EnvRun(config)


def register(id: str, entry_point: str, kwargs: Dict = {}, enable_assert: bool = True) -> None:
    """Register Env.

    Args:
        id (str): Identifier when calling Env. Must be unique.
        entry_point (str): Specify Env class. Must be a string that can be called with importlib.import_module.
        kwargs (Dict, optional): Arguments for Env instance. Defaults to {}.
        enable_assert (bool, optional): Throw assert if there is same id. Defaults to True.
    """
    global _registry

    if enable_assert:
        assert id not in _registry, f"{id} was already registered. entry_point={entry_point}"
    else:
        if id in _registry:
            logger.warn(f"{id} was already registered. Not registered. entry_point={entry_point}")
            return

    _registry[id] = {
        "entry_point": entry_point,
        "kwargs": kwargs,
    }

import logging
from typing import TYPE_CHECKING, Dict, Union

from srl.base.env.base import EnvBase
from srl.base.env.config import EnvConfig
from srl.base.exception import UndefinedError
from srl.utils.common import is_package_installed, load_module

if TYPE_CHECKING:
    from srl.base.env.env_run import EnvRun

logger = logging.getLogger(__name__)

_registry = {}


def make_base(config: Union[str, EnvConfig]) -> EnvBase:
    if isinstance(config, str):
        config = EnvConfig(config)

    env_name = config.name

    if env_name not in _registry:
        # --- srl内のenvはloadする
        if env_name in ["Grid", "EasyGrid", "Grid-layer"]:
            import srl.envs.grid  # noqa F401
        elif env_name == "IGrid":
            import srl.envs.igrid  # noqa F401
        elif env_name == "connectx":
            import srl.envs.connectx  # noqa F401
        elif env_name in ["OneRoad", "OneRoad-hard"]:
            import srl.envs.oneroad  # noqa F401
        elif env_name in [
            "Othello",
            "Othello6x6",
            "Othello4x4",
            "Othello-layer",
            "Othello6x6-layer",
            "Othello4x4-layer",
        ]:
            import srl.envs.othello  # noqa F401
        elif env_name in ["OX", "OX-layer"]:
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

    # --- load gym
    if env is None:
        _fgym = False
        if config.use_gym:
            if is_package_installed("gym"):
                _fgym = True
        else:
            if not is_package_installed("gymnasium"):
                _fgym = True

        if _fgym:
            import gym.error

            try:
                from srl.base.env.gym_wrapper import GymWrapper

                env = GymWrapper(config)
            except gym.error.NameNotFound as e:
                logger.warning(f"Gym failed to load. '{e}'")
            except gym.error.NamespaceNotFound as e:
                logger.warning(f"Gym failed to load. '{e}'")
            except Exception:
                raise
        else:
            import gymnasium.error

            try:
                from srl.base.env.gymnasium_wrapper import GymnasiumWrapper

                env = GymnasiumWrapper(config)
            except gymnasium.error.NameNotFound as e:
                logger.warning(f"Gymnasium failed to load. '{e}'")
            except gymnasium.error.NamespaceNotFound as e:
                logger.warning(f"Gymnasium failed to load. '{e}'")
            except Exception:
                raise

    if env is None:
        raise UndefinedError(f"'{env_name}' is not found.")

    return env


def make(config: Union[str, EnvConfig]) -> "EnvRun":
    from srl.base.env.env_run import EnvRun

    if isinstance(config, str):
        config = EnvConfig(config)
    return EnvRun(config)


def register(id: str, entry_point: str, kwargs: Dict = {}, check_duplicate: bool = True) -> None:
    """Register Env.

    Args:
        id (str): Identifier when calling Env. Must be unique.
        entry_point (str): Specify Env class. Must be a string that can be called with importlib.import_module.
        kwargs (Dict, optional): Arguments for Env instance. Defaults to {}.
        check_duplicate (bool, optional): Throw assert if there is same id. Defaults to True.
    """
    global _registry

    if check_duplicate:
        assert id not in _registry, f"{id} was already registered. entry_point={entry_point}"
    elif id in _registry:
        # 既にあれば上書き
        logger.warning(f"{id} was already registered, but I overwrote it. entry_point={entry_point}")

    _registry[id] = {
        "entry_point": entry_point,
        "kwargs": kwargs,
    }

import logging
import traceback
from typing import TYPE_CHECKING, Optional

from srl.utils.common import is_package_installed

if TYPE_CHECKING:
    import psutil


logger = logging.getLogger(__name__)

__enable_psutil: Optional[bool] = None
__psutil_process: Optional["psutil.Process"] = None


def read_memory() -> float:
    if not is_package_installed("psutil"):
        return -1
    import psutil

    return psutil.virtual_memory().percent


def read_cpu() -> float:
    global __enable_psutil, __psutil_process

    if __enable_psutil is None:
        __enable_psutil = False
        __psutil_process = None
        if is_package_installed("psutil"):
            try:
                import psutil

                __psutil_process = psutil.Process()
                __enable_psutil = True
            except Exception as e:
                logger.debug(traceback.format_exc())
                logger.info(e)

    if not __enable_psutil:
        return -1
    assert __psutil_process is not None

    import psutil

    return __psutil_process.cpu_percent(None) / psutil.cpu_count()

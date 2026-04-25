import logging
import traceback
from typing import TYPE_CHECKING, Optional

from srl.utils.common import is_package_installed

if TYPE_CHECKING:
    import psutil


logger = logging.getLogger(__name__)

__enable_psutil: Optional[bool] = None
__psutil_process: Optional["psutil.Process"] = None
__cpu_count: int = 1


def __setup_psutil():
    global __enable_psutil, __psutil_process, __cpu_count

    if __enable_psutil is not None:
        return

    __enable_psutil = False
    __psutil_process = None
    if is_package_installed("psutil"):
        try:
            import psutil

            proc = psutil.Process()
            proc.cpu_percent(None)  # 初回warmup
            __cpu_count = psutil.cpu_count() or 1
            __enable_psutil = True
            __psutil_process = proc
        except Exception as e:
            logger.debug(traceback.format_exc())
            logger.info(e)


def read_system_memory_percent() -> float:
    if not is_package_installed("psutil"):
        return -1
    import psutil

    return psutil.virtual_memory().percent


def read_process_memory_rss() -> int:
    global __enable_psutil, __psutil_process

    __setup_psutil()

    if not __enable_psutil:
        return -1
    assert __psutil_process is not None

    try:
        return __psutil_process.memory_info().rss
    except Exception as e:
        logger.debug(traceback.format_exc())
        logger.info(e)
        return -1


def read_cpu() -> float:
    """プロセスのCPU占有率を取得（全コア比）

    Returns:
        float: 0.0〜100.0（取得不可時は -1.0）
    """
    global __enable_psutil, __psutil_process, __cpu_count

    __setup_psutil()

    if not __enable_psutil:
        return -1
    assert __psutil_process is not None

    try:
        cpu = __psutil_process.cpu_percent(None)
        return cpu / __cpu_count
    except Exception as e:
        logger.debug(traceback.format_exc())
        logger.info(e)
        return -1

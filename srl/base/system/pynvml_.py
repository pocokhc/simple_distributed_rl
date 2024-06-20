import logging
import traceback
from typing import List, Optional, Tuple

from srl.utils.common import is_package_installed

logger = logging.getLogger(__name__)

# pynvmlはプロセス毎に管理
__used_pynvml: Optional[bool] = None


def setup_pynvml():
    global __used_pynvml
    if __used_pynvml is not None:
        return
    __used_pynvml = False
    if is_package_installed("pynvml"):
        try:
            import pynvml

            pynvml.nvmlInit()
            __used_pynvml = True

        except Exception as e:
            logger.debug(traceback.format_exc())
            logger.info(e)


def close_nvidia():
    global __used_pynvml
    if __used_pynvml is not None and __used_pynvml:
        __used_pynvml = None
        try:
            import pynvml

            pynvml.nvmlShutdown()
        except Exception:
            logger.info(traceback.format_exc())


def read_nvml() -> List[Tuple[int, float, float]]:
    setup_pynvml()
    if __used_pynvml is None:
        return []
    if not __used_pynvml:
        return []

    import pynvml

    gpu_num = pynvml.nvmlDeviceGetCount()
    gpus = []
    for device_id in range(gpu_num):
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpus.append((device_id, float(rate.gpu), float(rate.memory)))
    return gpus

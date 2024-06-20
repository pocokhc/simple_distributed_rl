import logging
import os
from typing import Tuple

logger = logging.getLogger(__name__)


__setup_device = False
__framework = ""
__used_device_tf = "/CPU"
__used_device_torch = "cpu"


def setup_device(
    framework: str,
    device: str,
    set_CUDA_VISIBLE_DEVICES_if_CPU: bool = True,
    tf_enable_memory_growth: bool = True,
    log_prefix: str = "",
) -> Tuple[str, str]:
    global __setup_device, __framework, __used_device_tf, __used_device_torch
    device = device.upper()

    # frameworkは "" の場合何もしない(フラグも立てない)
    if framework == "":
        return "/CPU", "cpu"
    if framework == "tf":
        framework = "tensorflow"
    assert framework in ["tensorflow", "torch"], "Framework can specify 'tensorflow' or 'torch'."

    if __setup_device:
        if __framework != framework:
            logger.warning(
                f"{log_prefix}Initialization with a different framework is not assumed. {__framework}!={framework}"
            )
        return __used_device_tf, __used_device_torch

    # logger
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"]
        logger.info(f"{log_prefix}[device] CUDA_VISIBLE_DEVICES='{cuda_devices}'")
    else:
        logger.info(f"{log_prefix}[device] CUDA_VISIBLE_DEVICES is not define.")

    # --- CUDA_VISIBLE_DEVICES ---
    if set_CUDA_VISIBLE_DEVICES_if_CPU:
        if "CPU" in device:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            logger.info(f"{log_prefix}[device] set CUDA_VISIBLE_DEVICES=-1")
        else:
            # CUDA_VISIBLE_DEVICES が -1 の場合のみ削除する
            if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "-1":
                del os.environ["CUDA_VISIBLE_DEVICES"]
                logger.info(f"{log_prefix}[device] del CUDA_VISIBLE_DEVICES")
    # -----------------------------

    # --- tf memory growth ---
    # Tensorflow,GPU がある場合に実施(CPUにしてもなぜかGPUの初期化は走る場合あり)
    if framework == "tensorflow" and tf_enable_memory_growth:
        try:
            import tensorflow as tf

            gpu_devices = tf.config.list_physical_devices("GPU")
            for d in gpu_devices:
                logger.info(f"{log_prefix}[device] (tf) set_memory_growth({d.name}, True)")
                tf.config.experimental.set_memory_growth(d, True)
        except Exception:
            s = f"{log_prefix}[device] (tf) 'set_memory_growth' failed."
            s += " Also consider 'Runner.setup_device(tf_enable_memory_growth=False)'."
            print(s)
            raise
    # -----------------------

    if "CPU" in device:
        # --- CPU ---
        if "CPU:" in device:
            t = device.split(":")
            used_device_tf = f"/CPU:{t[1]}"
            used_device_torch = f"cpu:{t[1]}"
        else:
            used_device_tf = "/CPU"
            used_device_torch = "cpu"
    else:
        used_device_tf = "/CPU"
        used_device_torch = "cpu"

        # --- GPU (AUTOの場合もあり) ---
        if framework == "tensorflow":
            # --- tensorflow GPU check
            import tensorflow as tf

            gpu_devices = tf.config.list_physical_devices("GPU")
            if len(gpu_devices) == 0:
                if "GPU" in device:
                    logger.warning(f"{log_prefix}[device] (tf) GPU is not found. {tf.config.list_physical_devices()}")

                used_device_tf = "/CPU"

            else:
                logger.info(f"{log_prefix}[device] (tf) gpu device: {len(gpu_devices)}")

                if "GPU:" in device:
                    t = device.split(":")
                    used_device_tf = f"/GPU:{t[1]}"
                else:
                    used_device_tf = "/GPU"

        if framework == "torch":
            # --- torch GPU check
            import torch

            if torch.cuda.is_available():
                logger.info(f"{log_prefix}[device] (torch) gpu device: {torch.cuda.get_device_name()}")

                if "GPU:" in device:
                    t = device.split(":")
                    used_device_torch = f"cuda:{t[1]}"
                else:
                    used_device_torch = "cuda"
            else:
                if "GPU" in device:
                    logger.warning(f"{log_prefix}[device] (torch) GPU is not found.")

                used_device_torch = "cpu"
    # -------------------------

    __setup_device = True
    __framework = framework
    __used_device_tf = used_device_tf
    __used_device_torch = used_device_torch

    logger.info(f"{log_prefix}[device] Initialized device. tf={used_device_tf}, torch={used_device_torch}")
    return used_device_tf, used_device_torch

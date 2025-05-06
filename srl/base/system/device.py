import logging
import os
import sys
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


__setup_device = False
__framework = ""
__used_device_tf = "/CPU"
__used_device_torch = "cpu"


def setup_device(
    framework: str,
    device: str,
    is_mp_main_process: Optional[bool] = None,
    set_CUDA_VISIBLE_DEVICES_if_CPU: bool = True,
    tf_enable_memory_growth: bool = True,
    tf_mixed_precision_policy_name: str = "",
    log_prefix: str = "",
) -> Tuple[str, str]:
    global __setup_device, __framework, __used_device_tf, __used_device_torch
    device = device.upper()

    # frameworkは "" の場合何もしない(フラグも立てない)
    if framework == "":
        return "/CPU", "cpu"
    assert framework in [
        "tensorflow",
        "torch",
    ], "Framework can specify 'tensorflow' or 'torch'."

    # 親プロセスでtorchがimportされていたら警告
    if (is_mp_main_process is not None) and is_mp_main_process:
        for key in sys.modules:
            if key == "torch":
                logger.warning("The parent process imports torch , which may lead to unexpected behavior with multiprocessing.")

    if __setup_device:
        if __framework != framework:
            logger.warning(f"{log_prefix}Initialization with a different framework is not assumed. {__framework}!={framework}")
        return __used_device_tf, __used_device_torch

    # logger
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"]
        logger.info(f"{log_prefix}[device] CUDA_VISIBLE_DEVICES='{cuda_devices}'")
    else:
        logger.info(f"{log_prefix}[device] CUDA_VISIBLE_DEVICES is not define.")

    # --- mp
    # tf   : spawnで起動、初期化は親プロセスのグローバルで実施
    # torch: spawnで起動、初期化は親プロセスでは実施せず、子プロセスでのみ実施する
    used_device_tf = "/CPU"
    used_device_torch = "cpu"
    if framework == "tensorflow":
        used_device_tf = _setup_tensorflow(
            device,
            is_mp_main_process,
            set_CUDA_VISIBLE_DEVICES_if_CPU,
            tf_enable_memory_growth,
            tf_mixed_precision_policy_name,
            log_prefix,
        )
    elif framework == "torch":
        used_device_torch = _setup_torch(
            device,
            is_mp_main_process,
            set_CUDA_VISIBLE_DEVICES_if_CPU,
            log_prefix,
        )

    __setup_device = True
    __framework = framework
    __used_device_tf = used_device_tf
    __used_device_torch = used_device_torch

    logger.info(f"{log_prefix}[device] Initialized device. tf={used_device_tf}, torch={used_device_torch}")
    return used_device_tf, used_device_torch


def _setup_tensorflow(
    device: str,
    is_mp_main_process: Optional[bool],
    set_CUDA_VISIBLE_DEVICES_if_CPU: bool,
    tf_enable_memory_growth: bool,
    tf_mixed_precision_policy_name: str,
    log_prefix: str,
):
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

    import tensorflow as tf

    # --- init GPU tensorflow
    # 内部で_initialize_physical_devices(初期化処理)が呼ばれる
    gpu_devices = tf.config.list_physical_devices("GPU")
    logger.info(f"{log_prefix}[device] initialize_physical_devices(GPU)")

    # --- tf memory growth ---
    # tf-gpuはCPUでもGPUがあるため実行
    # 新しいバージョンはCPUのみならエラーではなく[]がちゃんと返る
    if tf_enable_memory_growth:
        try:
            for d in gpu_devices:
                logger.info(f"{log_prefix}[device] (tf) set_memory_growth({d.name}, True)")
                tf.config.experimental.set_memory_growth(d, True)
        except Exception:
            s = f"{log_prefix}[device] (tf) 'set_memory_growth' failed."
            s += " Also consider 'Runner.setup_device(tf_enable_memory_growth=False)'."
            print(s)
            raise
    # -----------------------

    # --- tf Mixed precision ---
    if tf_mixed_precision_policy_name != "":
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy(tf_mixed_precision_policy_name)
        logger.info(f"{log_prefix}[device] (tf) set_global_policy({tf_mixed_precision_policy_name})")
    # -----------------------

    used_device_tf = "/CPU"
    if "CPU" in device:
        if "CPU:" in device:
            t = device.split(":")
            used_device_tf = f"/CPU:{t[1]}"
    elif len(gpu_devices) == 0:
        if "GPU" in device:
            logger.warning(f"{log_prefix}[device] (tf) GPU is not found. {tf.config.list_physical_devices()}")
        used_device_tf = "/CPU"
    else:  # GPU/AUTO check
        logger.info(f"{log_prefix}[device] (tf) gpu device: {len(gpu_devices)}")

        if "GPU:" in device:
            t = device.split(":")
            used_device_tf = f"/GPU:{t[1]}"
        else:
            used_device_tf = "/GPU"

    return used_device_tf


def _setup_torch(
    device: str,
    is_mp_main_process: Optional[bool],
    set_CUDA_VISIBLE_DEVICES_if_CPU: bool,
    log_prefix: str,
):
    # torchは子プロセスのみで初期化
    if (is_mp_main_process is not None) and is_mp_main_process:
        return "cpu"

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

    import torch

    used_device_torch = "cpu"
    if "CPU" in device:
        if "CPU:" in device:
            t = device.split(":")
            used_device_torch = f"cpu:{t[1]}"
    elif torch.cuda.is_available():
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
    return used_device_torch

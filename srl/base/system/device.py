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
    tf_mixed_precision_policy_name: str = "",
    log_prefix: str = "",
) -> Tuple[str, str]:
    global __setup_device, __framework, __used_device_tf, __used_device_torch
    device = device.upper()

    # frameworkは "" の場合何もしない(フラグも立てない)
    if framework == "":
        return "/CPU", "cpu"
    if framework == "tf":
        framework = "tensorflow"
    assert framework in [
        "tensorflow",
        "torch",
    ], "Framework can specify 'tensorflow' or 'torch'."

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

    # --- CUDA_VISIBLE_DEVICES ---
    # tf-gpuはCUDA_VISIBLE_DEVICESでGPUを消すと一定学習後プロセス自体が落ちる
    # 初期化時に "failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected" の出力あり
    # なのでtf-gpuはCUDA_VISIBLE_DEVICESで制御せず、tf.device側に任せる
    v216_older = False
    if framework == "tensorflow":
        import tensorflow as tf

        from srl.utils.common import compare_less_version

        v216_older = compare_less_version(tf.__version__, "2.16.0")
        if v216_older:
            set_CUDA_VISIBLE_DEVICES_if_CPU = False

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
    # tf-gpuはCPUでもGPUがあるため実行(上参照)
    # 新しいバージョンはCPUのみならエラーではなく[]がちゃんと返る
    if tf_enable_memory_growth and framework == "tensorflow":
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

    # --- tf Mixed precision ---
    if tf_mixed_precision_policy_name != "" and framework == "tensorflow":
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy(tf_mixed_precision_policy_name)
        logger.info(f"{log_prefix}[device] (tf) set_global_policy({tf_mixed_precision_policy_name})")
    # -----------------------

    used_device_tf = "/CPU"
    used_device_torch = "cpu"
    if "CPU" in device:
        if "CPU:" in device:
            t = device.split(":")
            used_device_tf = f"/CPU:{t[1]}"
            used_device_torch = f"cpu:{t[1]}"
    elif framework == "tensorflow":
        # --- tensorflow GPU/AUTO check
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

    elif framework == "torch":
        # --- torch GPU/AUTO check
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

    __setup_device = True
    __framework = framework
    __used_device_tf = used_device_tf
    __used_device_torch = used_device_torch

    logger.info(f"{log_prefix}[device] Initialized device. tf={used_device_tf}, torch={used_device_torch}")
    return used_device_tf, used_device_torch

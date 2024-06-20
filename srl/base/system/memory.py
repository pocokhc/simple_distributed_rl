import logging
import os
import traceback
from typing import Optional

from srl.utils.common import is_package_installed

logger = logging.getLogger(__name__)


def set_memory_limit(limit: Optional[int] = -1):
    """
    None : not change
    <=0  : auto
    int  : 指定サイズで設定
    """
    if limit is None:
        return
    if not is_package_installed("resource"):
        return

    try:
        # linux only
        import resource

        m_soft, m_hard = resource.getrlimit(resource.RLIMIT_DATA)

        # 設定されていない場合のみ設定
        if m_soft == -1 and m_hard == -1:
            limits = None
            if limit > 0:
                limits = (limit, -1)
            else:
                # auto, container上はcgroupで制限をかけてるらしいのでそれを見る
                path = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
                if os.path.isfile(path):
                    with open(path, "r") as fp:
                        memory_limit_in_bytes = int(fp.read())
                    limits = (memory_limit_in_bytes, -1)
                elif is_package_installed("psutil"):
                    # psutilで取得
                    import psutil

                    limits = (psutil.virtual_memory().total, -1)

            if limits is not None:
                logger.info(f"set resource.RLIMIT_DATA: {limits}")
                resource.setrlimit(resource.RLIMIT_DATA, limits)
            else:
                logger.info("not set resource.RLIMIT_DATA")
    except Exception:
        logger.info(traceback.format_exc())
        logger.warning("Failed to set resource.RLIMIT_DATA")

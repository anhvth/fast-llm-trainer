from loguru import logger
import os
def rank0_log_info(*args):
    if os.getenv("RANK", '0') == "0":
        logger.info(*args)


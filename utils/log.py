import logging
import os

def setup_logger(level=logging.INFO):
    logger = logging.getLogger('knowledge_system')
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

debug = os.environ.get('debug', 'False')
level = logging.INFO
if debug == 'True':
    level = logging.DEBUG

logger = setup_logger(level=level)
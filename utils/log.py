import logging
import os
from logging.handlers import TimedRotatingFileHandler
import os.path

def setup_logger(level=logging.INFO, log_file=None):
    logger = logging.getLogger('knowledge_system')
    logger.setLevel(level)
    
    # 控制台处理器
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # 文件处理器
    if log_file:
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # 创建按天分割的日志文件处理器，保留30天
        file_handler = TimedRotatingFileHandler(
            filename=log_file,
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

debug = os.environ.get('debug', 'False')
level = logging.INFO
if debug == 'True':
    level = logging.DEBUG

# 创建logs目录
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
log_file = os.path.join(log_dir, 'knowledge_system.log')

logger = setup_logger(level=level, log_file=log_file)
import logging
import os
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO):
    if log_file is None:
        log_file = f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else 'logs', exist_ok=True)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    
    return logger
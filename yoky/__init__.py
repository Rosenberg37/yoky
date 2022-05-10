import logging.config
import os
from pathlib import Path

# global variable
cache_root = Path(os.getenv('YOKY_CACHE_ROOT', Path(Path.home(), ".yoky")))
if not os.path.exists(cache_root):
    cache_root.mkdir()

module_root = os.path.dirname(__file__)

# logger
logger = logging.getLogger("yoky")
logger.setLevel(logging.INFO)
log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")

file_handler = logging.FileHandler(f"{cache_root}/runtime.log")
file_handler.setFormatter(log_format)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_format)
logger.addHandler(console_handler)

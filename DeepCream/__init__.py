import os
import logging

from DeepCream.deepcream import DeepCream as DeepCreamClass
from DeepCream.constants import (LOG_DIR,
                                 LOG_PATH,
                                 FILE_LOGGING_LEVEL,
                                 CONSOLE_LOGGING_LEVEL,
                                 FILE_LOGGING_FORMAT,
                                 CONSOLE_LOGGING_FORMAT,
                                 )

# Makes sure there is a log folder
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Start logging as soon as DeepCream is initialized

logger = logging.getLogger('DeepCream')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(LOG_PATH)
file_handler.setLevel(FILE_LOGGING_LEVEL)

console_handler = logging.StreamHandler()
console_handler.setLevel(CONSOLE_LOGGING_LEVEL)

file_handler.setFormatter(logging.Formatter(FILE_LOGGING_FORMAT))
console_handler.setFormatter(logging.Formatter(CONSOLE_LOGGING_FORMAT))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info('Initialised logger')


def initialize(directory, tpu_support=False):
    return DeepCreamClass(directory, tpu_support)

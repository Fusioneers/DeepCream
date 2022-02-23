import logging
import os

from DeepCream.constants import (LOG_DIR,
                                 DEBUG_LOG_PATH,
                                 INFO_LOG_PATH,
                                 CONSOLE_LOGGING_LEVEL,
                                 FILE_LOGGING_FORMAT,
                                 CONSOLE_LOGGING_FORMAT,
                                 )
from DeepCream.deepcream import DeepCream as DeepCreamClass

# Makes sure there is a log folder
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Start logging as soon as the DeepCream module is initialized
# There are 2 separate log files: the DEBUG... and the INFO... one.
# The DEBUG contains extensive information about all steps done by the program
# In the INFO the debug statements are missing and the format for each line is
# shorter to gain a quick overview of the runtime.
logger = logging.getLogger('DeepCream')
logger.setLevel(logging.DEBUG)

debug_file_handler = logging.FileHandler(DEBUG_LOG_PATH)
debug_file_handler.setLevel(logging.DEBUG)
debug_file_handler.setFormatter(logging.Formatter(FILE_LOGGING_FORMAT))

info_file_handler = logging.FileHandler(INFO_LOG_PATH)
info_file_handler.setLevel(logging.INFO)
info_file_handler.setFormatter(logging.Formatter(CONSOLE_LOGGING_FORMAT))

console_handler = logging.StreamHandler()
console_handler.setLevel(CONSOLE_LOGGING_LEVEL)
console_handler.setFormatter(logging.Formatter(CONSOLE_LOGGING_FORMAT))

logger.addHandler(debug_file_handler)
logger.addHandler(info_file_handler)
logger.addHandler(console_handler)

logger.info('Initialised logger')

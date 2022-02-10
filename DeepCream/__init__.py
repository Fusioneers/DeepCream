import logging
import os

from DeepCream.deepcream import DeepCream as DeepCreamClass
from DeepCream.constants import (LOGGING_FORMAT,
                                 LOGGING_LEVEL,
                                 LOG_DIR, LOG_PATH)

# Makes sure there is a log folder
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Start logging as soon as DeepCream is initialized
with open(LOG_PATH, 'w') as log:
    print('Opened log file: ' + str(LOG_PATH))
    logging.basicConfig(
        filename=LOG_PATH,
        format=LOGGING_FORMAT, level=LOGGING_LEVEL)
    print('Successfully configured logging')
    logging.info('Started DeepCream/__init__.py')


def initialize(directory):
    return DeepCreamClass(directory)

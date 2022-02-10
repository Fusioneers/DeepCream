import logging

from DeepCream.constants import (LOGGING_FORMAT,
                                 LOGGING_LEVEL,
                                 LOG_PATH,
                                 )

# TODO when folder not there, create one
with open(LOG_PATH, 'w') as log:
    print('Opened log file: ' + str(LOG_PATH))
    logging.basicConfig(
        filename=LOG_PATH,
        format=LOGGING_FORMAT, level=LOGGING_LEVEL)
    print('Successfully configured logging')
    logging.info('Started DeepCream/__init__.py')

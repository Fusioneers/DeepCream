import logging
from DeepCream.constants import (TIME_FORMAT,
                                 LOGGING_FORMAT,
                                 REP_PATH,
                                 LOGGING_LEVEL,
                                 LOG_PATH, )

with open(LOG_PATH, 'w') as log:
    print('opened LOG_PATH')
    logging.basicConfig(
        filename=LOG_PATH,
        format=LOGGING_FORMAT, level=LOGGING_LEVEL)
    print('configured logging')
    logging.info('Started DeepCream/__init__.py')

import logging
from DeepCream.constants import (time_format,
                                 logging_format,
                                 rep_path,
                                 logging_level,
                                 log_path, )

with open(log_path, 'w') as log:
    print('opened log_path')
    logging.basicConfig(
        filename=log_path,
        format=logging_format, level=logging_level,
        datefmt=time_format)
    print('configured logging')
    logging.info('Started DeepCream/__init__.py')

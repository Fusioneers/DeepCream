from DeepCream.__init__ import log_path
from DeepCream.constants import time_format, logging_format
import logging

with open(log_path, 'w') as log:
    logging.basicConfig(
        filename=log_path,
        format=logging_format, level=logging.DEBUG,
        datefmt=time_format)

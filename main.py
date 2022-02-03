import logging
import datetime
import os
from constants import time_format, logging_format

log_path = os.path.normpath(
    f'../DeepCream/logs/{datetime.datetime.today().strftime(time_format)}.log')

with open(log_path, 'w') as log:
    logging.basicConfig(
        filename=log_path,
        format=logging_format, level=logging.DEBUG, datefmt=time_format)

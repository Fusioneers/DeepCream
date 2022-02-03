import logging
from datetime import datetime
import os
from code.constants import time_format, logging_format

rep_path = os.path.normpath(os.path.dirname(__file__).removesuffix('code'))

log_path = os.path.normpath(os.path.join(
    rep_path, f'logs/{datetime.today().strftime(time_format)}.log'))

with open(log_path, 'w') as log:
    logging.basicConfig(
        filename=log_path,
        format=logging_format, level=logging.DEBUG,
        datefmt=time_format)

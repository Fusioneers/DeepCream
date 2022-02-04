import os
import logging
from datetime import datetime

rep_path = os.path.normpath(
    os.path.dirname(__file__).removesuffix('DeepCream'))

time_format = '%Y-%m-%d %H-%M-%S'

logging_format = '%(asctime)s: %(filename)s: line %(lineno)d: ' \
                 '%(levelname)s: %(message)s'

logging_level = logging.DEBUG

log_path = os.path.normpath(os.path.join(
    rep_path, f'logs/{datetime.today().strftime(time_format)}.log'))

default_appr_dist = 3
default_step_len = 2

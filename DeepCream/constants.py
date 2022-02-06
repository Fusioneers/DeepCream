import os
import logging
from datetime import datetime

REP_PATH = os.path.normpath(
    os.path.dirname(__file__).removesuffix('DeepCream'))

TIME_FORMAT = '%Y-%m-%d %H-%M-%S'

LOGGING_FORMAT = '%(asctime)s: %(name)s: %(filename)s: %(funcName)s: ' \
                 'line %(lineno)d: %(levelname)s: %(message)s'

LOGGING_LEVEL = logging.DEBUG

LOG_PATH = os.path.normpath(os.path.join(
    REP_PATH, f'logs/{datetime.today().strftime(TIME_FORMAT)}.log'))

DEFAULT_APPR_DIST = 3
DEFAULT_STEP_LEN = 2

import os
import logging
from datetime import datetime

ABS_PATH = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))

TIME_FORMAT = '%Y-%m-%d %H-%M-%S'

LOGGING_FORMAT = '%(asctime)s: %(name)s: %(filename)s: %(funcName)s: ' \
                 'line %(lineno)d: %(levelname)s: %(message)s'

LOGGING_LEVEL = logging.DEBUG

LOG_PATH = os.path.normpath(os.path.join(ABS_PATH, f'logs/{datetime.today().strftime(TIME_FORMAT)}.log'))

DEFAULT_APPR_DIST = 3
DEFAULT_STEP_LEN = 2
DEFAULT_BORDER_WIDTH = 70
DEFAULT_VAL_THRESHOLD = 30

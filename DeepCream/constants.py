import logging
import os
from datetime import datetime

ABS_PATH = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))

TIME_FORMAT = '%Y-%m-%d %H-%M-%S'

FILE_LOGGING_FORMAT = '%(asctime)s: %(name)s: %(funcName)s: ' \
                      'line %(lineno)d: %(levelname)s: %(message)s'

CONSOLE_LOGGING_FORMAT = '%(name)s: line %(lineno)d: %(levelname)s: %(' \
                         'message)s '

FILE_LOGGING_LEVEL = logging.DEBUG
CONSOLE_LOGGING_LEVEL = logging.WARNING

LOG_DIR = os.path.normpath(os.path.join(
    ABS_PATH, 'logs'))
LOG_PATH = os.path.normpath(os.path.join(
    LOG_DIR, f'{datetime.today().strftime(TIME_FORMAT)}.log'))

DEFAULT_APPR_DIST = 3
DEFAULT_STEP_LEN = 2
DEFAULT_BORDER_WIDTH = 70
DEFAULT_VAL_THRESHOLD = 30


# DEFAULT_COMPRESSION_DIM = ()


def get_time() -> str:
    return datetime.today().strftime(TIME_FORMAT)

import logging
import os
from datetime import datetime

DEBUG_MODE = True

ABS_PATH = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))

TIME_FORMAT = '%Y-%m-%d %H-%M-%S'


def get_time() -> str:
    return datetime.today().strftime(TIME_FORMAT)


FILE_LOGGING_FORMAT = '%(asctime)s: %(name)s: %(funcName)s: ' \
                      'line %(lineno)d: %(levelname)s: %(message)s'

CONSOLE_LOGGING_FORMAT = '%(name)s: line %(lineno)d: %(levelname)s: %(' \
                         'message)s '

CONSOLE_LOGGING_LEVEL = logging.DEBUG

log_time = get_time()
LOG_DIR = os.path.normpath(os.path.join(ABS_PATH, 'logs'))
DEBUG_LOG_PATH = os.path.join(LOG_DIR, f'DEBUG {log_time}.log')
INFO_LOG_PATH = os.path.join(LOG_DIR, f'INFO {log_time}.log')

DEFAULT_APPR_DIST = 3
DEFAULT_STEP_LEN = 2
DEFAULT_BORDER_WIDTH = 50
DEFAULT_VAL_THRESHOLD = 30

QUALITY_THRESHOLD = 0.2
MAX_DATABASE_SIZE = 1000 * ((2 ** 10) ** 2)

analysis_features = ['center x',
                     'center y',
                     'contour perimeter',
                     'contour area',
                     'hull perimeter',
                     'hull area',
                     'roundness',
                     'convexity',
                     'solidity',
                     'rectangularity',
                     'elongation',
                     'mean r',
                     'mean g',
                     'mean b',
                     'std r',
                     'std g',
                     'std b',
                     'std',
                     'transparency',
                     'sharp edges']

QUEUE_MAX_SIZE = 200
DEFAULT_DELAY = 0.1

import logging
import os
from datetime import datetime

# The path to the repository root i.e. .../DeepCream
ABS_PATH = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))

# The time format used in the modules
TIME_FORMAT = '%Y-%m-%d_%H-%M-%S'


# A shorthand for the current time represented as a string
def get_time() -> str:
    return datetime.today().strftime(TIME_FORMAT)


# The detailed logging format as used in DEBUG log
EXTENDED_LOGGING_FORMAT = '%(asctime)s: %(name)s: thread %(threadName)s: ' \
                          '%(funcName)s: line %(lineno)d: %(levelname)s: ' \
                          '%(message)s'

# A more concise logging format for the console or the INFO log
SHORT_LOGGING_FORMAT = '%(name)s: line %(lineno)d: %(levelname)s: %(' \
                       'message)s '

CONSOLE_LOGGING_LEVEL = logging.DEBUG

# The paths for the log files
log_time = get_time()
LOG_DIR = os.path.normpath(os.path.join(ABS_PATH, 'logs'))
DEBUG_LOG_PATH = os.path.join(LOG_DIR, f'DEBUG_{log_time}.log')
INFO_LOG_PATH = os.path.join(LOG_DIR, f'INFO_{log_time}.log')

# These are default values used by the Analysis class
DEFAULT_APPR_DIST = 3
DEFAULT_STEP_LEN = 2
DEFAULT_BORDER_WIDTH = 50
DEFAULT_VAL_THRESHOLD = 30

# This is the quality portion used in the database free_space procedure
QUALITY_THRESHOLD = 0.15

# The maximum size for the database
MAX_DATABASE_SIZE = 2900 * (10 ** 6)

# This is a list of features of a cloud returned by Analysis.evaluate
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

# The maximum queue size in DeepCream. A too high value causes a memory
# overflow.
QUEUE_MAX_SIZE = 5

# The delay used in the delay_supervisor and the main thread
DEFAULT_DELAY = 0.5

# The maximum time a single execution of a function in a thread is allowed to
# last, after which DeepCream is restarted
MAX_TIME = 60

# This is the number of invalid images after which the night mode takes place.
INVALID_ORIG_COUNT_THRESHOLD = 10

# The orig_priority at night
NIGHT_IMAGE_PRIORITY = -10

# If the OrigPrioritisationError is raised, the orig_priority is reduced by
# this amount.
ORIG_PRIORITISATION_ERROR_PENALTY = 20

# The amount per second by which the orig_priority returns to normal.
ORIG_PRIORITISATION_ERROR_COOLDOWN_RATE = 0.5 * DEFAULT_DELAY

# The temperature threshold above which the program is paused
TEMPERATURE_THRESHOLD = 95

# The duration in seconds the program is paused after a too high temperature
# occurred
TEMPERATURE_SLEEP = 60

# The maximum number of threads after which DeepCream is restarted
MAX_NUM_THREADS = 100

# Maximum time the program is allowed to run (in seconds)
runtime = 10800

# Time the program is going to run shorter than the runtime to ensure it
# finishes in time
buffer = 120

# The directory where the database is saved
directory = os.path.join(ABS_PATH, 'data', 'input')

# If this variable is true the program expects to have access to the
# coral TPU accelerator
tpu_support = False

# If this variable is set to true the program expects to be running on a
# Raspberry Pi with a connected Pi Camera
runs_on_pi = False

# The resolution of the camera connected to the astro pi
capture_resolution = (2592, 1952)

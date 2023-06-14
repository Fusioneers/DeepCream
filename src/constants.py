import logging
import os
from datetime import datetime

# The path to the repository root i.e. .../DeepCream
ABS_PATH = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))

CONSOLE_LOGGING_LEVEL = logging.WARNING

# These are default values used by the Analysis class
DEFAULT_APPR_DIST = 3
DEFAULT_STEP_LEN = 2
DEFAULT_BORDER_WIDTH = 50
DEFAULT_VAL_THRESHOLD = 30

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

# The maximum number of threads after which DeepCream is restarted
MAX_NUM_THREADS = 100

# The directory where the database is saved
directory = os.path.join(ABS_PATH, 'database')

import logging
import os.path

import DeepCream
from DeepCream.constants import logger, ABS_PATH

try:
    deepcream = DeepCream.initialize(os.path.join(ABS_PATH, 'data'))
    deepcream.start()
except BaseException as err:
    print(err)

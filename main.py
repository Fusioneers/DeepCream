import os.path

import DeepCream
from DeepCream.constants import ABS_PATH

try:
    deepcream = DeepCream.initialize(os.path.join(ABS_PATH, 'data/input'), os.path.join(ABS_PATH, 'data/output'))
    deepcream.start()
except BaseException as err:
    print(err)

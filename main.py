import time
import logging
import os.path
import DeepCream
from DeepCream.constants import ABS_PATH

start_time = time.time()

logger = logging.getLogger('DeepCream.main')

finished = False
runtime = 0.005  # time the program is allowed to run (in hours)

while int(time.time() - start_time) < (runtime * 60 * 60) and not finished:
    # Makes sure the DeepCream module keeps running for the whole time
    try:
        deepcream = DeepCream.initialize(os.path.join(ABS_PATH, 'data'), tpu_support=False)
        logger.info('Initialised DeepCream')
        allowed_execution_time = (runtime * 60 * 60) - (int(time.time() - start_time))
        logger.info('Calling DeepCream.run with ' + str(allowed_execution_time) + 's of execution time')
        deepcream.run(allowed_execution_time)
        finished = True
        logger.info('DeepCream execution time: ' + str(int(time.time() - start_time)) + 's')
    except Exception as err:
        logger.error(err)

end_time = time.time()

logger.info("Overall execution time: " + str(int(end_time - start_time)) + 's')
print("Overall execution time: " + str(int(end_time - start_time)) + 's')

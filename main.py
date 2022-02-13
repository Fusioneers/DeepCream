import time
import logging
import os.path
import DeepCream
from DeepCream.constants import ABS_PATH

start_time = time.time()

logger = logging.getLogger('DeepCream.main')

finished = False
runtime = 0.02  # time the program is supposed to run (in hours)

logger.info('Start time: ' + str(start_time))

while int(time.time() - start_time) < (runtime * 60 * 60) and not finished:
    # Makes sure the DeepCream module keeps running for the whole time
    try:
        logger.info('Initializing DeepCream')
        deepcream = DeepCream.initialize(os.path.join(ABS_PATH, 'data'), tpu_support=True)
        deepcream.run((runtime * 60 * 60) - (int(time.time() - start_time)))
        finished = True
        logger.info('DeepCream finished')
    except Exception as err:
        logging.error(err)

end_time = time.time()

logger.info('Start time: ' + str(end_time))
logger.info("Execution time: " + str(int(end_time - start_time)) + 's')
print("Execution time: " + str(int(end_time - start_time)) + 's')

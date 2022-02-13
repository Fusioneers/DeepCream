import logging
import time
import logging
import os.path
import DeepCream
from DeepCream.constants import logger, ABS_PATH

finished = False
runtime = 0.02  # time the program is supposed to run (in hours)
start_time = time.time()

logging.info('Start time: ' + str(start_time))

while int(time.time() - start_time) < (runtime * 60 * 60) and not finished:
    # Makes sure the DeepCream module keeps running for the whole time
    try:
        logging.info('Initializing DeepCream')
        deepcream = DeepCream.initialize(os.path.join(ABS_PATH, 'data'), tpu_support=True)
        deepcream.start()
        finished = True
        logging.info('DeepCream finished')
    except BaseException as err:
        logging.error(err)

end_time = time.time()

logging.info('Start time: ' + str(end_time))
logging.info("Execution time: " + str(int(end_time - start_time)) + 's')
print("Execution time: " + str(int(end_time - start_time)) + 's')

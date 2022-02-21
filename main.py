import logging
import os.path
import time
import traceback

import DeepCream
from DeepCream.constants import ABS_PATH

start_time = time.time()

logger = logging.getLogger('DeepCream.main')

finished = False
runtime = 180  # maximum time the program is allowed to run (in seconds)

while int(time.time() - start_time) < runtime and not finished:
    # Makes sure the DeepCream module keeps running for the whole time
    try:
        deepcream = DeepCream.initialize(
            os.path.join(ABS_PATH, 'data', 'input'),
            tpu_support=True, pi_camera=False, capture_resolution=(2592, 1952))
        logger.info('Initialised DeepCream')
        allowed_execution_time = runtime - (
            int(time.time() - start_time)) - 120
        logger.info(f'Calling DeepCream.run with {allowed_execution_time}s '
                    f'of execution time')
        deepcream.run(allowed_execution_time)
        finished = True
        logger.info('DeepCream execution time: ' + str(
            int(time.time() - start_time)) + 's')
    except MemoryError as err:
        logger.critical(
            'Stopping execution because database ran out of memory')
        finished = True
        logger.info(
            f'DeepCream execution time: {int(time.time() - start_time)}s')
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        logger.error(traceback.format_exc())

end_time = time.time()

logger.info(f'Overall execution time: {int(end_time - start_time)}s')
print(f'Overall execution time: {int(end_time - start_time)}s')

import logging
import os.path
import time

import DeepCream
from DeepCream.constants import ABS_PATH

start_time = time.time()

logger = logging.getLogger('DeepCream.main')

finished = False
# TODO convert to seconds
runtime = 0.005 * 60 * 60  # time the program is allowed to run (in hours)

while int(time.time() - start_time) < runtime and not finished:
    # Makes sure the DeepCream module keeps running for the whole time
    try:
        deepcream = DeepCream.initialize(os.path.join(ABS_PATH, 'data'),
                                         tpu_support=False)
        logger.info('Initialised DeepCream')
        allowed_execution_time = runtime - (
            int(time.time() - start_time))
        logger.info(f'Calling DeepCream.run with {allowed_execution_time}s '
                    f'of execution time')
        deepcream.run(allowed_execution_time)
        finished = True
        logger.info(
            f'DeepCream execution time: {int(time.time() - start_time)}s')
    except Exception as err:
        logger.error(err)
        # TODO Keyboard interrupt

end_time = time.time()

logger.info(f'Overall execution time: {int(end_time - start_time)}s')
print(f'Overall execution time: {int(end_time - start_time)}s')

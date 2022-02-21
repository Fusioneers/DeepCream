import logging
import os.path
import time

import DeepCream
from DeepCream.constants import ABS_PATH, DEFAULT_DELAY

# Gets the current time as start time
start_time = time.time()

logger = logging.getLogger('DeepCream.main')
finished = False
runtime = 10800  # Maximum time the program is allowed to run (in seconds)


def create_deepcream() -> DeepCream.deepcream.DeepCream:
    # Instances DeepCream
    deepcream = DeepCream.initialize(
        os.path.join(ABS_PATH, 'data', 'input'),
        tpu_support=False, pi_camera=False,
        capture_resolution=(2592, 1952))
    logger.info('Initialised DeepCream')

    deepcream.run()
    logger.info('Started DeepCream')
    
    return deepcream


# Loops as long as the three hours aren't over and the DeepCream module hasn't
# finished
deepcream = create_deepcream()
while time.time() - start_time < runtime and not finished:
    time.sleep(DEFAULT_DELAY)

    # Makes sure the DeepCream module keeps running for the whole time
    try:
        # Calculates the remaining execution time (minus 2 minutes as buffer)
        allowed_execution_time = runtime - time.time() + start_time - 120

        if not deepcream.alive and allowed_execution_time > 0:
            logger.warning(
                'DeepCream is not alive although the time is not up, '
                'attempting to reinstantiate DeepCream')
            deepcream = create_deepcream()
            # Starts the DeepCream module
            logger.info(
                f'Calling DeepCream.run with {allowed_execution_time}s '
                f'of execution time')

        if allowed_execution_time <= 0:
            # The allowed execution time is over, so the while loop can be
            # stopped
            finished = True
            deepcream.alive = False

            logger.info('DeepCream execution time: ' + str(
                int(time.time() - start_time)) + 's')
    except KeyboardInterrupt:
        pass

# Gets the current time as end time
end_time = time.time()

# Prints out and logs the overall execution time of the program Note:
# As mentioned in the documentation.md the threads might still be running at
# this point, but they are sure to run out before the 2 minutes of buffer are
# over
logger.info(f'Overall execution time: {int(end_time - start_time)}s')
print(f'Overall execution time: {int(end_time - start_time)}s')

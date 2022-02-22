import logging
import os.path
import time
import traceback

import DeepCream
from DeepCream.constants import ABS_PATH, DEFAULT_DELAY
from DeepCream.database import DataBase

# Gets the current time as start time
start_time = time.time()

logger = logging.getLogger('DeepCream.main')
finished = False
runtime = 200  # 10800  # Maximum time the program is allowed to run (in seconds)
buffer = 120  # Time the program is going to run shorter than the runtime to ensure it finishes in time


def create_deepcream() -> DeepCream.deepcream.DeepCream:
    """Instantiates DeepCream"""
    new_deepcream = DeepCream.initialize(
        os.path.join(ABS_PATH, 'data', 'input'),
        tpu_support=False, pi_camera=False,
        capture_resolution=(2592, 1952))
    logger.info('Initialised DeepCream')

    new_deepcream.run()
    logger.info('Started DeepCream')

    return new_deepcream


# Instances DeepCream for the first time
deepcream = create_deepcream()

# Keeps DeepCream alive as long as the three hours aren't over and the DeepCream module hasn't
# finished
while time.time() - start_time < runtime and not finished:
    time.sleep(DEFAULT_DELAY)

    # Makes sure the DeepCream module keeps running for the whole time
    try:
        # Calculates the remaining execution time (minus the buffer)
        allowed_execution_time = runtime - time.time() + start_time - buffer

        # If any exception makes it up to this level, the whole class is
        # restarted
        if not deepcream.alive and allowed_execution_time > 0:
            logger.warning(
                'DeepCream is not alive although the time is not up, '
                'attempting to reinstantiate DeepCream')
            deepcream = create_deepcream()
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
    except DataBase.DataBaseFullError as err:
        # If the program runs out of memory (because the 3GB are reached) the
        # while loop will stop before the three hours are over
        logger.critical(
            'Stopping execution because database ran out of memory')
        finished = True
        logger.info(
            f'DeepCream execution time: {int(time.time() - start_time)}s')
    except KeyboardInterrupt as e:
        logger.critical(traceback.format_exc())
        raise e from e
    except BaseException as e:
        # Catches all other exceptions and logs them
        logger.error(traceback.format_exc())
        deepcream.alive = False
    finally:
        if finished:
            deepcream.alive = False

# Gets the current time as end time
end_time = time.time()

# Prints out and logs the overall execution time of the program
# Note: As mentioned in the documentation.md the threads might still be running at
# this point, but they are sure to run out before the 2 minutes of buffer are
# over
logger.info(f'Overall execution time: {int(end_time - start_time)}s')
print(f'Overall execution time: {int(end_time - start_time)}s')

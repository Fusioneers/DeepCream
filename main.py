import logging
import os.path
import time
import traceback
import DeepCream
from DeepCream.constants import ABS_PATH

# Gets the current time as start time
start_time = time.time()

logger = logging.getLogger('DeepCream.main')
finished = False
runtime = 10800  # Maximum time the program is allowed to run (in seconds)

# Loops as long as the three hours aren't over and the DeepCream module hasn't finished
while int(time.time() - start_time) < runtime and not finished:
    # Makes sure the DeepCream module keeps running for the whole time
    try:
        # Instances DeepCream
        deepcream = DeepCream.initialize(
            os.path.join(ABS_PATH, 'data', 'input'),
            tpu_support=True, pi_camera=False, capture_resolution=(2592, 1952))
        logger.info('Initialised DeepCream')
        # Calculates the remaining execution time (minus 2 minutes as buffer)
        allowed_execution_time = runtime - (
            int(time.time() - start_time)) - 120
        if allowed_execution_time > 0:
            # Starts the DeepCream module
            logger.info(f'Calling DeepCream.run with {allowed_execution_time}s '
                        f'of execution time')
            deepcream.run(allowed_execution_time)
            # When the DeepCream module finished the loop will end before the three hours are over
            finished = True
            logger.info('DeepCream execution time: ' + str(
                int(time.time() - start_time)) + 's')
        else:
            # The allowed execution time is over, so the while loop can be stopped
            finished = True
            logger.info('DeepCream execution time: ' + str(
                int(time.time() - start_time)) + 's')
    except MemoryError as err:
        # If the program runs out of memory (because the 3GB are reached) the while loop will stop before the three
        # hours are over
        logger.critical(
            'Stopping execution because database ran out of memory')
        finished = True
        logger.info(
            f'DeepCream execution time: {int(time.time() - start_time)}s')
    except KeyboardInterrupt as e:
        # Catches and keyboard interrupts and halts the program
        raise e
    except Exception as e:
        # Catches all other exceptions and logs them
        logger.error(traceback.format_exc())

# Gets the current time as end time
end_time = time.time()

# Prints out and logs the overall execution time of the program Note: As mentioned in the documentation.md the
# threads might still be running at this point, but they are sure to run out before the 2 minutes of buffer are over
logger.info(f'Overall execution time: {int(end_time - start_time)}s')
print(f'Overall execution time: {int(end_time - start_time)}s')

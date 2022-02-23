"""A module containing a class DeepCream which manages threads to gain data
based on the camera in the Astro Pi. """

import logging
import os
import random
import threading as th
import time as t
import traceback
from queue import Queue
from typing import Callable

import cv2 as cv
import numpy as np

from DeepCream.classification.classification import Classification
from DeepCream.cloud_analysis.analysis import Analysis
from DeepCream.cloud_detection.cloud_detection import CloudDetection
from DeepCream.constants import (DEBUG_MODE,
                                 ABS_PATH,
                                 QUEUE_MAX_SIZE,
                                 get_time,
                                 DEFAULT_DELAY,
                                 MAX_TIME,
                                 INVALID_ORIG_COUNT_THRESHOLD,
                                 NIGHT_IMAGE_PRIORITY,
                                 ORIG_PRIORITISATION_ERROR_PENALTY,
                                 ORIG_PRIORITISATION_ERROR_COOLDOWN_RATE,
                                 )
from DeepCream.database import DataBase
from DeepCream.pareidolia.pareidolia import Pareidolia

logger = logging.getLogger('DeepCream.deepcream')

max_num_clouds = 15
max_border_proportion = 1


def thread(name: str) -> Callable:
    """A decorator factory for the threads used in the DeepCream class.

    This decorator runs the given function in a while loop, ensures that it
    stops if DeepCream is not alive and catches exceptions. It also sets
    DeepCream.alive to false if a function takes too long. Every iteration
    consists of a delay and an execution of the given function. The delay
    mechanism is explained in the delay_supervisor thread.

    Args:
        name:
        The name of the function to decorate e.g. 'save_orig'.

    Returns:
        A decorator for threads.
    """

    class timeout:
        """This class supervises the time a function takes to execute."""

        def __init__(self, deepcream, name, time):
            self.deepcream = deepcream
            self.name = name
            self.time = time
            self.exit = False

        def __enter__(self):
            self.th = th.Thread(target=self.callme, daemon=True)
            self.th.start()

        def callme(self):
            dtime = t.time()
            while t.time() - dtime < self.time and not self.exit:
                t.sleep(DEFAULT_DELAY)
            if not self.exit:
                logger.error(f'The function {name} took too long')
                self.deepcream.alive = False

        def __exit__(self, a, b, c):
            self.exit = True

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            logger.info(f'Started thread {name}')
            while self.alive:
                try:
                    t.sleep(getattr(self, f'_DeepCream__delay_{name}'))
                    dtime = t.time()
                    with timeout(self, name, MAX_TIME):
                        func(self, *args, **kwargs)
                    setattr(self, f'_DeepCream__duration_{name}',
                            (t.time() - dtime))
                except (DataBase.DataBaseFullError, SystemExit,
                        KeyboardInterrupt) as e:
                    # In case of a critical error, DeepCream is stopped
                    with self.lock:
                        logger.critical(traceback.format_exc())
                        self.alive = False
                        return
                except BaseException as e:
                    with self.lock:
                        logger.error(traceback.format_exc())
            logger.info(f'Finished thread {name}')

        return wrapper

    return decorator


class DeepCream:
    """A class to manage threads to retrieve and evaluate images from the
    camera.

    This class initialises a database and multiple threads which run each
    different parts of the program e.g. the taking of images or the saving of
    the classification.

    Attributes:
        directory:
        The directory in which the database is stored.

        database:
        The database where the results are stored.

        alive:
        A flag which indicates whether the current DeepCream is still 'alive'.
        During runtime, it is set to True. If it is set to False, all threads
        have time to finish their current cycle and stop then. This can happen
        if a critical error is encountered, the time is up or a function takes
        too long to complete.

        orig_priority:
        The priority of taking images. A value of 0 means no specific trend.
        A high value urges the program to take more pictures, negative values
        slow that down. More specifically, if it is below 0, the delay of
        get_orig is set to orig_priority ** 2. If it is below 0, the penalty
        decays over time.

        invalid_orig_rate:
        An indicator for nighttime. Everytime an image is classified as invalid
        by the review_orig thread, this value goes up by 1, and otherwise
        decreases by 1 down to 0. If it reaches a threshold, the orig_priority
        is set to a negative constant so that more images are analysed.
        If there are more valid images the orig_priority rises again.

        camera:
        The camera connected to the pi.

        capture_resolution:
        The resolution of the images taken by the camera.

        cloud_detection:
        The cloud detection instance used by the program.

        classification:
        The classification instance used by the program.

        pareidolia:
        The pareidolia instance used by the program.

        lock:
        A threading lock to ensure that some operations are not interrupted by
        the threading mechanism. This is used for database access and some
        logging.

        ...queue:
        These are queues to communicate between threads. Every output by a
        producer thread (get_orig, orig_review, get_mask,
        get_analysis_pareidolia and get_classification) is sent into one of
        the queues. The objects in the queues are then taken one by one from on
        of the consumer threads (review_orig, save_orig, save_mask,
        save_analysis, save_pareidolia and save_classification) and processed
        further or saved into the database.

        ...th...:
        The thread objects used by the program. The threads are explained more
        detailed in their respective functions.

        ...delay...:
        The amount of time each function waits before each execution.

        ...duration...:
        The amount of time each execution in the threads took the last time.

    """

    def __init__(self, directory: str, tpu_support: bool = False,
                 pi_camera: bool = False, capture_resolution=(2592, 1952)):
        """Initialises DeepCream.

        Args:
            directory:
            The directory in which the database is stored.

            tpu_support:
            Whether a tpu is connected to the astro pi.

            pi_camera:
            Whether a camera is connected to the astro pi.

            capture_resolution:
            The resolution of the images taken by the camera. This has only an
            effect if pi_camera is True.

        """

        logger.debug('Attempting to initialise DeepCream')
        self.directory = directory

        if DEBUG_MODE:
            self.database = DataBase(
                os.path.join(ABS_PATH, 'data', f'database_{get_time()}'))
        else:
            self.database = DataBase(
                os.path.join(ABS_PATH, 'data', 'database'))

        self.alive = True

        self.orig_priority = 0

        self.invalid_orig_rate = 0

        if pi_camera:
            try:
                from picamera import PiCamera
                self.camera = PiCamera()
                self.camera.resolution = capture_resolution
                self.camera.framerate = 15
            except (
                    DataBase.DataBaseFullError, SystemExit,
                    KeyboardInterrupt) as e:
                logger.critical(e)
                return
            except Exception as e:
                logger.error('Camera not configured: ', str(e))
                raise ValueError('Camera not configured')
        else:
            self.camera = None
        self.capture_resolution = capture_resolution

        self.cloud_detection = CloudDetection(tpu_support=tpu_support)
        self.classification = Classification()
        self.pareidolia = Pareidolia(tpu_support=False)

        self.lock = th.Lock()

        self.orig_review_queue = Queue(maxsize=QUEUE_MAX_SIZE)
        self.orig_queue = Queue(maxsize=QUEUE_MAX_SIZE)
        self.mask_queue = Queue(maxsize=QUEUE_MAX_SIZE)
        self.analysis_queue = Queue(maxsize=QUEUE_MAX_SIZE)
        self.classification_queue = Queue(maxsize=QUEUE_MAX_SIZE)
        self.pareidolia_queue = Queue(maxsize=QUEUE_MAX_SIZE)
        logger.debug('Initialised queues')

        self.__th_delay_supervisor = th.Thread(target=self.__delay_supervisor,
                                               daemon=True)

        self.__th_get_orig = th.Thread(target=self.__get_orig, daemon=True)
        self.__th_review_orig = th.Thread(target=self.__review_orig,
                                          daemon=True)
        self.__th_get_mask = th.Thread(target=self.__get_mask, daemon=True)
        self.__th_get_analysis_pareidolia = th.Thread(
            target=self.__get_analysis_pareidolia,
            daemon=True)
        self.__th_get_classification = th.Thread(
            target=self.__get_classification, daemon=True)

        self.__th_save_orig = th.Thread(target=self.__save_orig, daemon=True)
        self.__th_save_mask = th.Thread(target=self.__save_mask, daemon=True)
        self.__th_save_analysis = th.Thread(target=self.__save_analysis,
                                            daemon=True)
        self.__th_save_classification = th.Thread(
            target=self.__save_classification, daemon=True)
        self.__th_save_pareidolia = th.Thread(target=self.__save_pareidolia,
                                              daemon=True)

        self.__th_delay_supervisor.name = 'delay_supervisor'

        self.__th_get_orig.name = 'get_orig'
        self.__th_review_orig.name = 'review_orig'
        self.__th_get_mask.name = 'get_mask'
        self.__th_get_analysis_pareidolia.name = 'get_analysis_pareidolia'
        self.__th_get_classification.name = 'get_classification'

        self.__th_save_orig.name = 'save_orig'
        self.__th_save_mask.name = 'save_mask'
        self.__th_save_analysis.name = 'save_analysis'
        self.__th_save_classification.name = 'save_classification'
        self.__th_save_pareidolia.name = 'save_pareidolia'

        logger.debug('Initialised threads')

        self.__delay_get_orig = 1
        self.__delay_review_orig = 1
        self.__delay_get_mask = 1
        self.__delay_get_analysis_pareidolia = 1
        self.__delay_get_classification = 1

        self.__delay_save_orig = 1
        self.__delay_save_mask = 1
        self.__delay_save_analysis = 1
        self.__delay_save_classification = 1
        self.__delay_save_pareidolia = 1

        self.__duration_get_orig = 1
        self.__duration_review_orig = 1
        self.__duration_get_mask = 1
        self.__duration_get_analysis_pareidolia = 1
        self.__duration_get_classification = 1

        self.__duration_save_orig = 1
        self.__duration_save_mask = 1
        self.__duration_save_analysis = 1
        self.__duration_save_classification = 1
        self.__duration_save_pareidolia = 1

        logger.info('Initialisation of DeepCream finished')

    def run(self):
        """Starts the threads."""

        logger.debug('Attempting to start running')

        self.__th_delay_supervisor.start()

        t.sleep(DEFAULT_DELAY)

        self.__th_get_orig.start()
        self.__th_review_orig.start()
        self.__th_get_mask.start()
        self.__th_get_analysis_pareidolia.start()
        self.__th_get_classification.start()

        self.__th_save_orig.start()
        self.__th_save_mask.start()
        self.__th_save_analysis.start()
        self.__th_save_classification.start()
        self.__th_save_pareidolia.start()
        logger.debug('Started threads')

    def __delay_supervisor(self):
        """Manages the delays of the threads.

        Some executions in threads take longer than others. For example the
        taking of images can happen way faster than the generation of the
        masks. An instability is not desired because every image needs to
        undergo all threads and unclassified images do more harm than good.
        The delay_supervisor adds and updates delay to the threads so that
        the overall frequency of the threads is the same.
        """

        def get_delay(start, end):
            start_delay = getattr(self, f'_DeepCream__delay_{start}')
            start_duration = getattr(self, f'_DeepCream__duration_{start}')
            end_duration = getattr(self, f'_DeepCream__duration_{end}')
            delay = start_delay + start_duration - end_duration
            if delay >= 0:
                return start_delay, delay
            else:
                new_start_delay = end_duration - start_duration
                return new_start_delay, 0

        def adjust_delays():
            # orig based threads
            self.__delay_get_orig, self.__delay_review_orig = get_delay(
                'get_orig', 'review_orig')
            self.__delay_review_orig, self.__delay_save_orig = get_delay(
                'review_orig', 'save_orig')
            delay_save_orig_a, self.__delay_get_mask = get_delay(
                'save_orig', 'get_mask')
            delay_save_orig_b, delay_get_analysis_pareidolia_a = get_delay(
                'save_orig', 'get_analysis_pareidolia')
            self.__delay_save_orig = max(delay_save_orig_a,
                                         delay_save_orig_b)

            # mask based threads
            self.__delay_get_mask, self.__delay_save_mask = get_delay(
                'get_mask', 'save_mask')
            self.__delay_save_mask, \
            delay_get_analysis_pareidolia_b = get_delay(
                'save_mask', 'get_analysis_pareidolia')
            self.__delay_save_mask *= 0.75

            # analysis based threads
            self.__delay_get_analysis_pareidolia, \
            self.__delay_save_analysis = get_delay(
                'get_analysis_pareidolia', 'save_analysis')
            self.__delay_get_analysis_pareidolia = min(
                delay_get_analysis_pareidolia_a,
                delay_get_analysis_pareidolia_b)
            self.__delay_save_analysis, self.__delay_get_classification \
                = get_delay('save_analysis', 'get_classification')
            self.__delay_get_classification, \
            self.__delay_save_classification = get_delay(
                'get_classification', 'save_classification')

            # pareidolia based threads
            self.__delay_get_analysis_pareidolia, \
            self.__delay_save_pareidolia = get_delay(
                'get_analysis_pareidolia', 'save_pareidolia')

        def check_orig_priority():
            if self.orig_priority < 0:
                self.orig_priority += ORIG_PRIORITISATION_ERROR_COOLDOWN_RATE

                self.__delay_get_orig = self.orig_priority ** 2
                self.__delay_review_orig = self.orig_priority ** 2
                self.__delay_save_orig = self.orig_priority ** 2
                self.__delay_save_mask = 0
                self.__delay_get_analysis_pareidolia = 0
                self.__delay_save_analysis = 0
                self.__delay_save_pareidolia = 0
                self.__delay_get_classification = 0
                self.__delay_save_classification = 0
                logger.debug('Adjusted delays')

        def check_invalid_orig_count():
            if self.invalid_orig_rate > INVALID_ORIG_COUNT_THRESHOLD:
                logger.warning('There are a lot of invalid images, attempting '
                               'to adjust delays')
                self.orig_priority = NIGHT_IMAGE_PRIORITY
            if self.invalid_orig_rate == 0:
                self.orig_priority = 0

        while self.alive:
            try:
                t.sleep(DEFAULT_DELAY)
                adjust_delays()
                check_orig_priority()
                check_invalid_orig_count()

            except (DataBase.DataBaseFullError, SystemExit,
                    KeyboardInterrupt) as e:
                with self.lock:
                    logger.critical(traceback.format_exc())
                    self.alive = False
                    return
            except BaseException as e:
                with self.lock:
                    logger.error(traceback.format_exc())

    @thread('get_orig')
    def __get_orig(self):
        if self.camera is not None:
            orig = np.empty(
                (self.capture_resolution[1], self.capture_resolution[0], 3),
                dtype=np.uint8)
            self.camera.capture(orig, 'rgb')
        else:
            # Returns a random (RGB) image (placeholder until real camera)
            random_file_name = random.choice(os.listdir(self.directory))
            orig = cv.cvtColor(
                cv.imread(os.path.join(self.directory, random_file_name)),
                cv.COLOR_BGR2RGB)

        self.orig_review_queue.put(orig)
        logger.debug('Got orig')
        logger.debug(
            f'Current get_orig time is '
            f'{self.__delay_get_orig + self.__duration_get_orig}'
            f' seconds per image')

    @thread('review_orig')
    def __review_orig(self):
        if not self.orig_review_queue.empty():
            logger.debug('Reviewing orig image')
            orig = self.orig_review_queue.get()
            gray = cv.cvtColor(orig, cv.COLOR_RGB2GRAY)

            # The values are set so that bad images certainly will be filtered
            # out, but that might also lead to good images being filtered out
            if cv.mean(gray)[0] < 55 or cv.mean(gray)[0] > 125:
                out = False
            else:
                out = True

            if out:
                self.orig_queue.put(orig)

                if self.invalid_orig_rate > 0:
                    self.invalid_orig_rate -= 1
            else:
                if not self.invalid_orig_rate > INVALID_ORIG_COUNT_THRESHOLD:
                    self.invalid_orig_rate += 1

    @thread('save_orig')
    def __save_orig(self):
        if not self.orig_queue.empty():
            orig = self.orig_queue.get()
            try:
                with self.lock:
                    self.database.save_orig(orig)
            except DataBase.OrigPrioritisationError as e:
                logger.error(e)
                self.orig_priority -= ORIG_PRIORITISATION_ERROR_PENALTY

    @thread('get_mask')
    def __get_mask(self):
        with self.lock:
            identifier = self.database.load_id('orig creation time',
                                               'created mask')
        if identifier is not None:
            with self.lock:
                orig = self.database.load_orig(identifier)
            mask = self.cloud_detection.evaluate_image(orig).astype(
                'uint8') * 255
            if not np.count_nonzero(mask):
                logger.warning(
                    'There are no clouds on the image, attempting to '
                    f'delete orig {identifier}')
                self.database.delete_orig(identifier)
                self.invalid_orig_rate += 1
            self.mask_queue.put((mask, identifier))

    @thread('save_mask')
    def __save_mask(self):
        if not self.mask_queue.empty():
            mask, identifier = self.mask_queue.get()
            with self.lock:
                self.database.save_mask(mask, identifier)
            logger.debug(f'Saved mask to {identifier}')

    @thread('get_analysis_pareidolia')
    def __get_analysis_pareidolia(self):
        with self.lock:
            identifier = self.database.load_id('created mask',
                                               'created analysis')

        if identifier is not None:
            with self.lock:
                orig = self.database.load_orig(identifier)
            mask = None
            try:
                with self.lock:
                    mask = self.database.load_mask(identifier)
            except ValueError as err:
                logger.error(err)

            if mask is not None:
                analysis = Analysis(orig, mask, max_num_clouds,
                                    max_border_proportion)

                if not analysis.clouds:
                    logger.warning('There are no valid clouds on the image,'
                                   f' attempting to delete orig {identifier}')
                    self.database.delete_orig(identifier)
                    self.invalid_orig_rate += 1

                df = analysis.evaluate()
                self.analysis_queue.put((df, identifier))

                masks = [np.tile(cloud.mask[:, :, np.newaxis], 3)
                         for cloud in analysis.clouds]
                pareidolia = self.pareidolia.evaluate_clouds(masks)
                self.pareidolia_queue.put((pareidolia, identifier))

    @thread('save_analysis')
    def __save_analysis(self):
        if not self.analysis_queue.empty():
            analysis, identifier = self.analysis_queue.get()
            with self.lock:
                self.database.save_analysis(analysis, identifier)

    @thread('save_pareidolia')
    def __save_pareidolia(self):
        if not self.pareidolia_queue.empty():
            pareidolia, identifier = self.pareidolia_queue.get()
            with self.lock:
                self.database.save_pareidolia(pareidolia, identifier)

    @thread('get_classification')
    def __get_classification(self):
        with self.lock:
            identifier = self.database.load_id('created analysis',
                                               'created classification')
        if identifier is not None:
            with self.lock:
                analysis = self.database.load_analysis(identifier)
            classification = self.classification.evaluate(
                analysis)
            self.classification_queue.put((classification, identifier))

    @thread('save_classification')
    def __save_classification(self):
        if not self.classification_queue.empty():
            classification, identifier = self.classification_queue.get()
            with self.lock:
                self.database.save_classification(classification,
                                                  identifier)

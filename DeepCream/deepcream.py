import logging
import os
import random
import threading as th
import time as t
from queue import Queue

import cv2 as cv
import numpy as np
import cv2

from DeepCream.classification.classification import Classification
from DeepCream.cloud_analysis.analysis import Analysis
from DeepCream.cloud_detection.cloud_detection import CloudDetection
from DeepCream.constants import (DEBUG_MODE,
                                 ABS_PATH,
                                 QUEUE_MAX_SIZE,
                                 get_time,
                                 DEFAULT_DELAY,
                                 )
from DeepCream.database import DataBase

logger = logging.getLogger('DeepCream.deepcream')

max_num_clouds = 15
max_border_proportion = 1


class DeepCream:
    def __init__(self, directory: str, tpu_support: bool = False,
                 pi_camera: bool = False, capture_resolution=(2560, 1920)):
        logger.debug('Attempting to initialise DeepCream')
        self.directory = directory

        self.alive = True
        self.lock = th.Lock()

        if pi_camera:
            try:
                from picamera import PiCamera
                self.camera = PiCamera()
                self.camera.resolution = capture_resolution
                self.camera.framerate = 15
            except KeyboardInterrupt as e:
                logger.error(e)
                raise KeyboardInterrupt from e
            except Exception as e:
                logger.error('Camera not configured: ', str(e))
                raise ValueError('Camera not configured')
        else:
            self.camera = None
        self.capture_resolution = capture_resolution

        self.cloud_detection = CloudDetection(tpu_support=tpu_support)
        self.classification = Classification()
        if DEBUG_MODE:
            self.database = DataBase(
                os.path.join(ABS_PATH, 'data', f'database {get_time()}'))
        else:
            self.database = DataBase(
                os.path.join(ABS_PATH, 'data', 'database'))

        self.orig_review_queue = Queue(maxsize=QUEUE_MAX_SIZE)
        self.orig_queue = Queue(maxsize=QUEUE_MAX_SIZE)
        self.mask_queue = Queue(maxsize=QUEUE_MAX_SIZE)
        self.analysis_queue = Queue(maxsize=QUEUE_MAX_SIZE)
        self.classification_queue = Queue(maxsize=QUEUE_MAX_SIZE)
        self.pareidolia_queue = Queue(maxsize=QUEUE_MAX_SIZE)
        logger.debug('Initialised queues')

        self.__th_get_orig = th.Thread(target=self.__get_orig, daemon=True)
        self.__th_review_orig = th.Thread(target=self.__review_orig,
                                          daemon=True)
        self.__th_get_mask = th.Thread(target=self.__get_mask, daemon=True)
        self.__th_get_analysis = th.Thread(target=self.__get_analysis,
                                           daemon=True)
        self.__th_get_classification = th.Thread(
            target=self.__get_classification, daemon=True)
        self.__th_get_pareidolia = th.Thread(target=self.__get_pareidolia,
                                             daemon=True)

        self.__th_save_orig = th.Thread(target=self.__save_orig, daemon=True)
        self.__th_save_mask = th.Thread(target=self.__save_mask, daemon=True)
        self.__th_save_analysis = th.Thread(target=self.__save_analysis,
                                            daemon=True)
        self.__th_save_classification = th.Thread(
            target=self.__save_classification, daemon=True)
        self.__th_save_pareidolia = th.Thread(target=self.__save_pareidolia,
                                              daemon=True)
        logger.debug('Initialised threads')

        self.__delay_get_orig = 0
        self.__delay_review_orig = 0
        self.__delay_get_mask = 0
        self.__delay_get_analysis = 0
        self.__delay_get_classification = 0
        self.__delay_get_pareidolia = 0

        self.__delay_save_orig = 0
        self.__delay_save_mask = 0
        self.__delay_save_analysis = 0
        self.__delay_save_classification = 0
        self.__delay_save_pareidolia = 0

        self.__duration_get_orig = 0
        self.__duration_review_orig = 0
        self.__duration_get_mask = 0
        self.__duration_get_analysis = 0
        self.__duration_get_classification = 0
        self.__duration_get_pareidolia = 0

        self.__duration_save_orig = 0
        self.__duration_save_mask = 0
        self.__duration_save_analysis = 0
        self.__duration_save_classification = 0
        self.__duration_save_pareidolia = 0

        logger.info('Initialisation of DeepCream finished')

    def run(self, allowed_execution_time: int):
        logger.debug('Attempting to start running')

        start_time = t.time()

        self.__th_get_orig.start()
        self.__th_review_orig.start()
        self.__th_get_mask.start()
        self.__th_get_analysis.start()
        self.__th_get_classification.start()
        self.__th_get_pareidolia.start()

        self.__th_save_orig.start()
        self.__th_save_mask.start()
        self.__th_save_analysis.start()
        self.__th_save_classification.start()
        self.__th_save_pareidolia.start()
        logger.debug('Started threads')

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

        while int(t.time() - start_time) < allowed_execution_time:
            t.sleep(DEFAULT_DELAY)
            # print(self.__delay_get_mask + self.__duration_get_mask
            #       - self.__delay_save_mask - self.__delay_save_mask)

            # TODO instead of controlling speed wait til a threshold is reached

            # first branch
            self.__delay_get_orig, self.__delay_review_orig = get_delay(
                'get_orig', 'review_orig')
            self.__delay_review_orig, self.__delay_save_orig = get_delay(
                'review_orig', 'save_orig')
            delay_save_orig_a, self.__delay_get_mask = get_delay(
                'save_orig', 'get_mask')
            delay_save_orig_b, delay_get_analysis_a = get_delay(
                'save_orig', 'get_analysis')
            self.__delay_save_orig = max(delay_save_orig_a, delay_save_orig_b)

            # second branch
            self.__delay_get_mask, self.__delay_save_mask = get_delay(
                'get_mask', 'save_mask')
            delay_save_mask_a, delay_get_analysis_b = get_delay(
                'save_mask', 'get_analysis')
            delay_save_mask_b, self.__delay_get_pareidolia = get_delay(
                'save_mask', 'get_pareidolia')
            self.__delay_save_mask = min(delay_save_mask_a, delay_save_mask_b)

            # third branch
            self.__delay_get_analysis, self.__delay_save_analysis = get_delay(
                'get_analysis', 'save_analysis')
            self.__delay_get_analysis = min(delay_get_analysis_a,
                                            delay_get_analysis_b)
            self.__delay_save_analysis, \
            self.__delay_get_classification = get_delay('save_analysis',
                                                        'get_classification')
            self.__delay_get_classification, \
            self.__delay_save_classification = get_delay('get_classification',
                                                         'save_classification')

            # fourth branch
            self.__delay_get_pareidolia, \
            self.__delay_save_pareidolia = get_delay('get_pareidolia',
                                                     'save_pareidolia')

        self.alive = False
        logger.info('Finished running')

    def thread(name: str):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                logger.info(f'Started thread {name}')
                while self.alive:
                    t.sleep(getattr(self, f'_DeepCream__delay_{name}'))
                    dtime = t.time()
                    func(self, *args, **kwargs)
                    setattr(self, f'_DeepCream__duration_{name}',
                            (t.time() - dtime))
                logger.info(f'Finished thread {name}')

            return wrapper

        return decorator

    @thread('get_orig')
    def __get_orig(self):
        if self.camera:
            orig = np.empty(
                (self.capture_resolution[1], self.capture_resolution[0], 3))
            self.camera.capture(orig, 'rgb')
        else:
            # Returns a random (RGB) image (placeholder until real camera)
            random_file_name = random.choice(os.listdir(self.directory))
            orig = cv.cvtColor(
                cv.imread(os.path.join(self.directory, random_file_name)),
                cv.COLOR_BGR2RGB)

        self.orig_review_queue.put(orig)
        logger.debug('Got orig')

    @thread('review_orig')
    def __review_orig(self):
        if not self.orig_review_queue.empty():
            logger.debug('Reviewing orig image')
            orig = self.orig_review_queue.get()
            gray = cv.cvtColor(orig, cv.COLOR_RGB2GRAY)

            # TODO find ideal values
            if cv.mean(gray)[0] < 20 or cv.mean(gray)[0] > 150:
                out = False
            # TODO take the average of the four corners (bigger than 1px)
            elif gray[0][0] > 20:
                out = False
            else:
                out = True

            if out:
                self.orig_queue.put(orig)

    @thread('save_orig')
    def __save_orig(self):
        if not self.orig_queue.empty():
            orig = self.orig_queue.get()
            with self.lock:
                self.database.save_orig(orig)

    @thread('get_mask')
    def __get_mask(self):
        with self.lock:
            identifier = self.database.load_orig_id_by_empty_mask()
        if identifier is not None:
            with self.lock:
                orig = self.database.load_orig(identifier)
            mask = self.cloud_detection.evaluate_image(orig).astype(
                'uint8') * 255
            self.mask_queue.put((mask, identifier))

    @thread('save_mask')
    def __save_mask(self):
        if not self.mask_queue.empty():
            mask, identifier = self.mask_queue.get()
            with self.lock:
                self.database.save_mask(mask, identifier)
            logger.debug(f'Saved mask to {identifier}')

    @thread('get_analysis')
    def __get_analysis(self):
        with self.lock:
            identifier = self.database.load_orig_id_by_empty_analysis()
        if identifier is not None:
            with self.lock:
                orig = self.database.load_orig(identifier)
            try:
                with self.lock:
                    mask = self.database.load_mask(identifier)

                analysis = Analysis(orig, mask, max_num_clouds,
                                    max_border_proportion)
                df = analysis.evaluate()
                self.analysis_queue.put((df, identifier))

            except ValueError as err:
                logger.error(err)

    @thread('save_analysis')
    def __save_analysis(self):
        if not self.analysis_queue.empty():
            analysis, identifier = self.analysis_queue.get()
            with self.lock:
                self.database.save_analysis(analysis, identifier)

    @thread('get_classification')
    def __get_classification(self):
        with self.lock:
            identifier = self.database. \
                load_analysis_id_by_empty_classification()
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

    @thread('get_pareidolia')
    def __get_pareidolia(self):
        pass

    @thread('save_pareidolia')
    def __save_pareidolia(self):
        if not self.pareidolia_queue.empty():
            pareidolia, identifier = self.pareidolia_queue.get()
            with self.lock:
                self.database.save_pareidolia(pareidolia, identifier)

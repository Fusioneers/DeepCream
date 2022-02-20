import logging
import os
import random
import threading as th
import time as t
from queue import Queue

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
    def __init__(self, directory: str, tpu_support: bool = False, pi_camera: bool = False, capture_resolution=(2560, 1920)):
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


        # TODO __review_photo is missing from the threads?
        self.orig_queue = Queue(maxsize=QUEUE_MAX_SIZE)
        self.mask_queue = Queue(maxsize=QUEUE_MAX_SIZE)
        self.analysis_queue = Queue(maxsize=QUEUE_MAX_SIZE)
        self.classification_queue = Queue(maxsize=QUEUE_MAX_SIZE)
        self.pareidolia_queue = Queue(maxsize=QUEUE_MAX_SIZE)
        logger.debug('Initialised queues')

        self.__th_get_orig = th.Thread(target=self.__get_orig, daemon=True)
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

        self.__th_get_orig.name = 'Get_orig'
        self.__th_get_mask.name = 'Get_mask'
        self.__th_get_analysis.name = 'Get_analysis'
        self.__th_get_classification.name = 'Get_classification'
        self.__th_get_pareidolia.name = 'Get_pareidolia'

        self.__th_save_orig.name = 'Save_orig'
        self.__th_save_mask.name = 'Save_mask'
        self.__th_save_analysis.name = 'Save_analysis'
        self.__th_save_classification.name = 'Save_classification'
        self.__th_save_pareidolia.name = 'Save_pareidolia'

        logger.info('Initialisation of DeepCream finished')

    def run(self, allowed_execution_time: int):
        logger.debug('Attempting to start running')

        start_time = t.time()

        self.__th_get_orig.start()
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

        while int(t.time() - start_time) < allowed_execution_time:
            # TODO add variable delay in threads to balance queues
            t.sleep(DEFAULT_DELAY)
        self.alive = False
        logger.info('Finished running')

    def __review_photo(self, image: np.ndarray) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # TODO find ideal values
        if cv2.mean(gray)[0] < 20 or cv2.mean(gray)[0] > 150:
            return False

        # TODO take the average of the four corners (bigger than 1px)
        if gray[0][0] > 20:
            return False

        return True

    def __get_orig(self):
        logger.info('Started thread get_orig')
        while self.alive:
            t.sleep(15)
            if self.camera:
                orig = np.empty((self.capture_resolution[1], self.capture_resolution[0], 3), dtype=np.uint8)
                self.camera.capture(orig, 'rgb')
            else:
                # Returns a random (RGB) image (placeholder until real camera)
                random_file_name = random.choice(os.listdir(self.directory))
                orig = cv2.cvtColor(
                    cv2.imread(os.path.join(self.directory, random_file_name)),
                    cv2.COLOR_BGR2RGB)

            self.orig_queue.put(orig)
            logger.debug('Got orig')

    def __save_orig(self):
        logger.info('Started thread save_orig')
        while self.alive:
            if not self.orig_queue.empty():
                orig = self.orig_queue.get()
                with self.lock:
                    self.database.save_orig(orig)
        logger.info('Finished thread save_orig')

    def __get_mask(self):
        logger.info('Started thread get_mask')
        while self.alive:
            with self.lock:
                identifier = self.database.load_orig_id_by_empty_mask()
            if identifier is not None:
                with self.lock:
                    orig = self.database.load_orig(identifier)
                mask = self.cloud_detection.evaluate_image(orig).astype(
                    'uint8') * 255
                self.mask_queue.put((mask, identifier))

        logger.info('Finished thread get_mask')

    def __save_mask(self):
        logger.info('Started thread save_mask')
        while self.alive:
            if not self.mask_queue.empty():
                mask, identifier = self.mask_queue.get()
                with self.lock:
                    self.database.save_mask(mask, identifier)
                logger.debug(f'Saved mask to {identifier}')
        logger.info('Finished thread save_mask')

    def __get_analysis(self):
        logger.info('Started thread get_analysis')
        while self.alive:
            t.sleep(DEFAULT_DELAY)
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

    logger.info('Finished thread get_analysis')

    def __save_analysis(self):
        logger.info('Started thread save_analysis')
        while self.alive:
            if not self.analysis_queue.empty():
                analysis, identifier = self.analysis_queue.get()
                with self.lock:
                    self.database.save_analysis(analysis, identifier)
        logger.info('Finished thread save_analysis')

    def __get_classification(self):
        logger.info('Started thread evaluate')
        while self.alive:
            t.sleep(DEFAULT_DELAY)
            with self.lock:
                identifier = self.database. \
                    load_analysis_id_by_empty_classification()
            if identifier is not None:
                with self.lock:
                    analysis = self.database.load_analysis(identifier)
                classification = self.classification.evaluate(
                    analysis)
                self.classification_queue.put((classification, identifier))

        logger.info('Finished thread evaluate')

    def __save_classification(self):
        logger.info('Started thread save_classification')
        while self.alive:
            if not self.classification_queue.empty():
                classification, identifier = self.classification_queue.get()
                with self.lock:
                    self.database.save_classification(classification,
                                                      identifier)
        logger.info('Finished thread save_classification')

    def __get_pareidolia(self):
        logger.info('Started thread get_pareidolia')
        while self.alive:
            t.sleep(DEFAULT_DELAY)
        logger.info('Finished thread get_pareidolia')

    def __save_pareidolia(self):
        logger.info('Started thread save_pareidolia')
        while self.alive:
            if not self.pareidolia_queue.empty():
                pareidolia, identifier = self.pareidolia_queue.get()
                with self.lock:
                    self.database.save_pareidolia(pareidolia, identifier)
        logger.info('Finished thread save_pareidolia')

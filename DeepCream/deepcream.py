import logging
import os
import random
import threading as th
import time
from queue import Queue

import cv2

from DeepCream.classification.classification import Classification
from DeepCream.cloud_analysis.analysis import Analysis
from DeepCream.cloud_detection.cloud_detection import CloudDetection
from DeepCream.constants import ABS_PATH, queue_max_size
from DeepCream.database import DataBase

logger = logging.getLogger('DeepCream.deepcream')

max_num_clouds = 15
max_border_proportion = 1


class DeepCream:
    def __init__(self, directory: str, tpu_support: bool):
        self.directory = directory

        self.alive = True
        self.lock = th.Lock()

        self.cloud_detection = CloudDetection(tpu_support=tpu_support)
        self.classification = Classification()
        self.database = DataBase(os.path.join(ABS_PATH, 'database'))

        self.orig_queue = Queue(maxsize=queue_max_size)
        self.mask_queue = Queue(maxsize=queue_max_size)
        self.analysis_queue = Queue(maxsize=queue_max_size)
        self.classification_queue = Queue(maxsize=queue_max_size)
        self.paradolia_queue = Queue(maxsize=queue_max_size)

        self.__th_get_orig = th.Thread(target=self.__get_orig(), daemon=True)
        self.__th_get_mask = th.Thread(target=self.__get_mask(), daemon=True)
        self.__th_get_analysis = th.Thread(target=self.__get_analysis(),
                                           daemon=True)
        self.__th_get_classification = th.Thread(
            target=self.__get_classification(), daemon=True)
        self.__th_get_paradolia = th.Thread(target=self.__get_pareidolia(),
                                            daemon=True)

    def run(self, allowed_execution_time: int):
        print(allowed_execution_time)

        start_time = time.time()

        self.__th_get_orig.start()
        self.__th_get_mask.start()
        self.__th_get_analysis.start()
        self.__th_get_classification.start()
        self.__th_get_paradolia.start()

        while int(time.time() - start_time) < allowed_execution_time:
            # TODO add variable delay in threads to balance queues
            time.sleep(1)
        self.alive = False

    def __get_orig(self):
        logger.info('started thread get_orig')
        while self.alive:
            logger.info('Take photo')
            # Returns a random (RGB) image (placeholder until real camera)
            random_file_name = random.choice(os.listdir(self.directory))
            orig = cv2.cvtColor(
                cv2.imread(os.path.join(self.directory, random_file_name)),
                cv2.COLOR_BGR2RGB)

            self.orig_queue.put(orig)

    def __save_orig(self):
        logger.info('Started thread save_orig')
        while self.alive:
            if not self.orig_queue.empty():
                orig = self.orig_queue.get()
                with self.lock:
                    self.database.save_orig(orig)

    def __get_mask(self):
        logger.info('Started thread get_mask')
        while self.alive:
            with self.lock:
                orig, identifier = self.database.load_orig_by_empty_mask()

            mask = self.cloud_detection.evaluate_image(orig)
            self.mask_queue.put((mask, identifier))

    def __save_mask(self):
        logger.info('started thread save_mask')
        while self.alive:
            if not self.mask_queue.empty():
                mask, identifier = self.mask_queue.get()
                with self.lock:
                    self.database.save_mask(mask, identifier)

    def __get_analysis(self):
        logger.info('Started thread get_analysis')
        while self.alive:
            with self.lock:
                orig, identifier = self.database.load_orig_by_empty_analysis()

            with self.lock:
                mask = self.database.load_mask_by_id(identifier)

            analysis = Analysis(orig, mask, max_num_clouds,
                                max_border_proportion)
            df = analysis.evaluate()
            self.analysis_queue.put(df)

    def __save_analysis(self):
        logger.info('Started thread save_analysis')
        while self.alive:
            if not self.analysis_queue.empty():
                analysis, identifier = self.analysis_queue.get()
                with self.lock:
                    self.database.save_analysis(analysis, identifier)

    def __get_classification(self):
        logger.info('Started thread get_classification')
        while self.alive:
            with self.lock:
                analysis, identifier = \
                    self.database.load_analysis_by_empty_classification()

            classification = self.classification.get_classification(analysis)
            self.classification_queue.put(classification)

    def __save_classification(self):
        while self.alive:
            if not self.classification_queue.empty():
                classification, identifier = self.classification_queue.get()
                with self.lock:
                    self.database.save_classification(classification,
                                                      identifier)

    def __get_pareidolia(self):
        while self.alive:
            pass

    def __save_pareidolia(self):
        while self.alive:
            if not self.paradolia_queue.empty():
                pareidolia, identifier = self.paradolia_queue.get()
                with self.lock:
                    self.database.save_paradolia(pareidolia, identifier)

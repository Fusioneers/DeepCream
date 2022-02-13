import logging
import os

import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm
from DeepCream.cloud_analysis.analysis import Analysis

from DeepCream.cloud_detection.cloud_detection import CloudDetection
from DeepCream.constants import ABS_PATH

input_dir = os.path.join(ABS_PATH, 'data\\input')

output_dir = os.path.join(ABS_PATH, 'cloud_imgs')

cloud_detection = CloudDetection()

num_img = len(os.listdir(input_dir))

for i, path in tqdm(enumerate(os.scandir(input_dir)), total=num_img):
    logging.info(os.path.normpath(path.path))
    img = cv.cvtColor(cv.imread(os.path.normpath(path.path)), cv.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    mask = cloud_detection.evaluate_image(img)

    analysis = Analysis(img, mask, 5, 0.5)
    for n, cloud in enumerate(analysis.clouds):
        cv.imwrite(
            os.path.normpath(
                os.path.join(output_dir, f'img_{i}_cloud_{n}.png')),
            cloud.img)
    if i == 5:
        break

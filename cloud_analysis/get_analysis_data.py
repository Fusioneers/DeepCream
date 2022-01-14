import os
import time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from cloud_analysis import Analysis

path = os.path.realpath(__file__).removesuffix(r'cloud_analysis\get_analysis_data.py')
print(path)

# TODO get data for a lot of images and compare features for hopefully meaningful results

orig_path = path + r'sample_data\Data\zz_astropi_1_photo_364.jpg'
num_clouds = 5
distance = 20
num_angles = 4


def plot(img):
    cv.imshow('', cv.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4))))
    cv.waitKey(0)
    cv.destroyAllWindows()


orig = cv.imread(orig_path)
plot(orig)

dtime = time.time()

analysis = Analysis(orig_path, num_clouds, distance, num_angles)

print(analysis.shape)
print(analysis.texture)

print(f'computation time:  {time.time() - dtime}')

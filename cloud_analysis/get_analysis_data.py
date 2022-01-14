import numpy as np
import matplotlib.pyplot as plt
import os
import time
import cv2 as cv
from cloud_analysis import Analysis

path = os.path.realpath(__file__).removesuffix(r'cloud_analysis\get_analysis_data.py')

# TODO get data for a lot of images and compare features for hopefully meaningful results


img_path = path + r'sample_data\Data\zz_astropi_1_photo_364.jpg'
num_clouds = 5
distance = 20
num_angles = 16


def plot(img):
    cv.imshow('', cv.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4))))
    cv.waitKey(0)
    cv.destroyAllWindows()


dtime = time.time()

analysis = Analysis(img_path, num_clouds, distance, num_angles)
print('finished constructing analysis')

print('\n################################\n')
for cloud in analysis.clouds:
    print(analysis.clouds[0].shape)
    print('\n--------------------------------\n')
    print(analysis.clouds[0].texture)
    print('\n################################\n')

print(f'computation time:  {time.time() - dtime}')

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import seaborn as sns
import os
import time
import cv2 as cv
from cloud_analysis import Analysis

path = os.path.realpath(__file__).removesuffix(r'cloud_analysis\get_analysis_data.py')

# TODO get data for a lot of images and compare features for hopefully meaningful results


img_path = path + r'sample_data\Data\zz_astropi_1_photo_352.jpg'
num_clouds = 5
distance = 20
num_glcm = 16
c_dist = 5
img = cv.imread(img_path)


def plot(img):
    plt.imshow(cv.cvtColor(img, cv.COLOR_RGB2BGR), cmap='gray')
    plt.show()


dtime = time.time()

analysis = Analysis(cv.imread(img_path), num_clouds, distance, num_glcm, c_dist)

if True:
    print('\n################################\n')
    for cloud in analysis.clouds:
        image = cloud.img
        plot(image)

        print(cloud.shape)
        print('\n--------------------------------\n')
        print(cloud.texture)
        print(f'    edge width: {cloud.edge_width()}')
        print('\n################################\n')

        plt.imshow(cloud.texture.contrast_img, cmap='gray')
        plt.show()


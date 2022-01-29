import os

import cv2 as cv
import matplotlib.pyplot as plt

from analysis import Analysis


def plot(img):
    plt.imshow(cv.cvtColor(img, cv.COLOR_RGB2BGR))
    plt.show()


path = os.path.realpath(__file__).removesuffix(r'cloud_analysis\analysis_test.py')

analysis = Analysis(cv.imread(path + r'sample_data\Data\zz_astropi_1_photo_364.jpg'), 5, 0.1)


def info(n):
    cloud = analysis.clouds[n]
    # print(cloud.shape)
    return cloud.mean_diff_edges(10, 50, 50)
    # plot(cloud.img)


for n in range(5):
    var = info(n)
    print('')

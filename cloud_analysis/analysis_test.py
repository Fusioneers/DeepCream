from analysis import Analysis
import os
import cv2 as cv
import matplotlib.pyplot as plt


def plot(img):
    plt.imshow(cv.cvtColor(img, cv.COLOR_RGB2BGR))
    plt.show()


path = os.path.realpath(__file__).removesuffix(r'cloud_analysis\analysis_test.py')

analysis = Analysis(cv.imread(path + r'sample_data\Data\zz_astropi_1_photo_364.jpg'), 5)


def info(n):
    cloud = analysis.clouds[n]
    print(cloud.shape)
    print(cloud.edges(50, 100, 500))
    plot(cloud.img)


info(0)
info(1)
info(-1)

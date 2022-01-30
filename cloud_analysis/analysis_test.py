import cv2 as cv
import matplotlib.pyplot as plt

from analysis import Analysis


def plot(img):
    plt.imshow(cv.cvtColor(img, cv.COLOR_RGB2BGR))
    plt.show()


image = cv.imread('../sample_data/Data/zz_astropi_1_photo_364.jpg')
analysis = Analysis(image, 5, 0.1)


def info(i):
    cloud = analysis.clouds[i]
    return cloud.mean_diff_edges(10, 50, 50)
    # plot(cloud.img)


for n in range(5):
    print(info(n))
    print('')

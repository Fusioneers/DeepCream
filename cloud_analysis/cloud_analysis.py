import os
import time
import cv2 as cv
import numpy as np
import skimage
import matplotlib.pyplot as plt

path = os.path.realpath(__file__).removesuffix(r'\cloud_analysis\cloud_analysis.py')

# Constants
NUM_CLOUDS = 5
DISTANCE = 20
NUM_ANGLES = 8
ANGLES = np.arange(0, 2 * np.pi, np.pi / NUM_ANGLES * 2)


# for more detailed explanation of the outputs of the methods
# see http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth06.pdf
# and http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf

def plot(img):
    cv.imshow('', cv.resize(img, (int(orig.shape[1] / 4), int(orig.shape[0] / 4))))
    cv.waitKey(0)
    cv.destroyAllWindows()


class Cloud:

    def __init__(self, orig, contour):
        self.contour = contour
        self.contour_perimeter = cv.arcLength(self.contour, True)
        self.contour_area = cv.contourArea(self.contour)
        self.hull = cv.convexHull(self.contour)
        self.hull_perimeter = cv.arcLength(self.hull, True)
        self.hull_area = cv.contourArea(self.hull)

        self.orig = orig
        self.mask = np.zeros(self.orig.shape[:2], np.uint8)

        cv.drawContours(self.mask, [self.contour], 0, (255, 255, 255), -1)

        self.img = cv.bitwise_and(self.orig, self.orig, mask=self.mask)
        self.img_grey = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

        # graylevel co-ocurrence matrix
        self.glcm = skimage.feature.graycomatrix(self.img_grey, [DISTANCE], ANGLES, normed=True)[:, :, 0, :]
        self.glcm = np.mean(self.glcm, axis=2)
        self.glcm = self.glcm[1:, 1:]

        # greylevel distance statistics
        self.glds = [np.sum(self.glcm.diagonal(n) + np.sum(self.glcm.diagonal(-n))) for n in range(256)]
        self.glds = self.glds / np.sum(self.glds)

    def get_circularity(self):
        return (4 * np.pi * self.contour_area) / (self.hull_perimeter ** 2)

    def get_rectangularity(self):
        (x, y), (width, height), angle = cv.minAreaRect(self.contour)
        return self.contour_area / (width * height)

    def get_convexity(self):
        return self.hull_perimeter / self.contour_perimeter

    def get_compactness(self):
        return (4 * np.pi * self.contour_area) / (self.contour_perimeter ** 2)

    def get_solidity(self):
        return self.contour_area / self.hull_area

    def get_elongation(self):
        (x, y), (width, height), angle = cv.minAreaRect(self.contour)
        return min(width, height) / max(width, height)

    def get_mean(self):
        return cv.mean(self.img, mask=self.mask)

    def get_standard_deviation(self):
        _, stddev = cv.meanStdDev(self.img, mask=self.mask)
        return stddev

    def get_glcm_contrast(self):
        i, j = np.indices(self.glcm.shape)
        coefficients = ((i - j) ** 2).astype(int)
        return np.sum(coefficients * self.glcm)

    def get_glds_median(self):
        return np.std(self.glds)

    def get_transparency(self):
        pass

    def get_edges(self):
        pass

    def analyze_cloud(self):
        pass


def rescale_image(mask, orig):
    height, width, channels = orig.shape
    mask_re = cv.resize(mask, (width, height))
    mask_re = cv.cvtColor(mask_re, cv.COLOR_BGR2GRAY)
    return mask_re


def get_contours(img):
    img = cv.medianBlur(img, 3)
    contours, _ = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    areas = [cv.contourArea(cnt) for cnt in contours]
    max_areas = np.sort(areas)[-NUM_CLOUDS:]
    new_contours = [contours[np.where(areas == max_area)[0][0]] for max_area in max_areas]
    return new_contours


if __name__ == '__main__':
    # cv.namedWindow("output", cv.WINDOW_NORMAL)
    dtime = time.time()
    orig = cv.imread(path + '/sample_data/Data/zz_astropi_1_photo_364.jpg')
    # cf = CloudFilter()
    # mask, color_image = cf.evaluate_image(path + '/sample_data/Data/zz_astropi_1_photo_364.jpg')
    mask = cv.imread('mask_re_364.jpg')
    mask_re = rescale_image(mask, orig)
    contours = get_contours(mask_re)
    clouds = [Cloud(orig, contour) for contour in contours]

    # print(time.time() - dtime)
    print('\n#######################################\n')
    for cloud in clouds:
        print('Shape Analysis:\n')
        print(f'contour area: {cloud.contour_area}')
        print(f'hull area: {cloud.hull_area}')
        print(f'circularity: {cloud.get_circularity()}')
        print(f'rectangularity: {cloud.get_rectangularity()}')
        print(f'convexity: {cloud.get_convexity()}')
        print(f'compactness: {cloud.get_compactness()}')
        print(f'solidity: {cloud.get_solidity()}')
        print(f'elongation: {cloud.get_elongation()}')
        print('\n---------------------------------------\n')
        print('Texture Analysis:\n')
        print(f'mean: {cloud.get_mean()}')
        print(f'standard deviation: {cloud.get_standard_deviation()}')
        print(f'glcm contrast:\n{cloud.get_glcm_contrast()}')
        print(f'glds median:\n{cloud.get_glds_median()}')
        print('\n#######################################\n')
        plt.plot(range(len(cloud.glds)), cloud.glds,
                 label='relative occurrence within cloud')
        plt.plot(range(len(np.diff(cloud.glds))), np.diff(cloud.glds),
                 label='difference of relative occurrences within cloud')
        plt.title('Greylevel Distance Statistics')
        plt.xlabel('greyscale distance')
        plt.legend()
        plt.show()
        plot(cloud.img)

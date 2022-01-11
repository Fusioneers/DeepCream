import os
import time
import cv2 as cv
import numpy as np

path = os.path.realpath(__file__).removesuffix(r'\cloud_analysis\cloud_analysis.py')

NUM_CLOUDS = 5


def plot(img):
    cv.imshow('', cv.resize(img, (int(orig.shape[1] / 4), int(orig.shape[0] / 4))))
    cv.waitKey(0)
    cv.destroyAllWindows()


class Cloud:

    def __init__(self, orig, contour):
        self.width, self.height, self.channels = orig.shape

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

        # self.plot(self.mask)
        # self.plot(self.orig)
        # self.plot(self.img)

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

    def get_texture(self):
        pass

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


def get_contours(img, NUM_CLOUDS):
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
    contours = get_contours(mask_re, NUM_CLOUDS)
    clouds = [Cloud(orig, contour) for contour in contours]

    # print(time.time() - dtime)
    for cloud in clouds:
        print('#######################')
        print(f'circularity: {cloud.get_circularity()}')
        print(f'rectangularity: {cloud.get_rectangularity()}')
        print(f'convexity: {cloud.get_convexity()}')
        print(f'compactness: {cloud.get_compactness()}')
        print(f'solidity: {cloud.get_solidity()}')
        print(f'elongation: {cloud.get_elongation()}')
        plot(cloud.img)

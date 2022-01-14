import cv2 as cv
import numpy as np
import skimage
from cloud_detection.cloud_filter import CloudFilter


# for more detailed explanation of the outputs of the methods
# see http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth06.pdf
# and http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf


class Analysis:

    def __init__(self, orig_path, num_clouds, distance, num_angles):
        self.orig = cv.imread(orig_path)

        self.height, self.width, self.channels = self.orig.shape

        cloud_filter = CloudFilter()
        mask, _ = cloud_filter.evaluate_image(orig_path)
        mask_re = cv.resize(mask, (self.width, self.height))
        mask_re = cv.cvtColor(mask_re, cv.COLOR_BGR2GRAY)

        all_contours, _ = cv.findContours(cv.medianBlur(mask_re, 3), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        areas = [cv.contourArea(cnt) for cnt in all_contours]
        max_areas = np.sort(areas)[-num_clouds:]
        self.contours = [all_contours[np.where(areas == max_area)[0][0]] for max_area in max_areas]

        self.clouds = [self.Cloud(self.orig, contour, distance, num_angles) for contour in self.contours]

    class Cloud:
        def __init__(self, orig, contour, distance, num_angles):
            self.orig = orig
            self.contour = contour
            self.mask = np.zeros(self.orig.shape[:2], np.uint8)
            cv.drawContours(self.mask, [self.contour], 0, (255, 255, 255), -1)

            self.img = cv.bitwise_and(self.orig, self.orig, mask=self.mask)

            self.form = self.Shape(self.contour)
            self.texture = self.Texture(self.img, self.mask, distance, num_angles)

        class Shape:

            def __init__(self, contour):
                self.contour = contour
                self.contour_perimeter = cv.arcLength(self.contour, True)
                self.contour_area = cv.contourArea(self.contour)
                self.hull = cv.convexHull(self.contour)
                self.hull_perimeter = cv.arcLength(self.hull, True)
                self.hull_area = cv.contourArea(self.hull)

            def __str__(self):
                print(f'Shape Analysis:')
                print(f'    contour perimeter: {self.contour_perimeter}')
                print(f'    hull perimeter: {self.hull_perimeter}')
                print(f'    contour area: {self.contour_area}')
                print(f'    hull area: {self.hull_area}')
                print(f'    circularity: {self.get_circularity()}')
                print(f'    rectangularity: {self.get_rectangularity()}')
                print(f'    convexity: {self.get_convexity()}')
                print(f'    compactness: {self.get_compactness()}')
                print(f'    solidity: {self.get_solidity()}')
                print(f'    elongation: {self.get_elongation()}')

            def get_circularity(self):
                return (4 * np.pi * self.contour_area) / (self.hull_perimeter ** 2)

            def get_rectangularity(self):
                _, (width, height), angle = cv.minAreaRect(self.contour)
                return self.contour_area / (width * height)

            def get_convexity(self):
                return self.hull_perimeter / self.contour_perimeter

            def get_compactness(self):
                return (4 * np.pi * self.contour_area) / (self.contour_perimeter ** 2)

            def get_solidity(self):
                return self.contour_area / self.hull_area

            def get_elongation(self):
                _, (width, height), angle = cv.minAreaRect(self.contour)
                return min(width, height) / max(width, height)

        class Texture:
            def __init__(self, img, mask, distance, num_angles):
                self.img = img
                self.mask = mask
                self.img_grey = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

                # graylevel co-ocurrence matrix
                angles = np.arange(0, 2 * np.pi, np.pi / num_angles * 2)
                self.glcm = skimage.feature.graycomatrix(self.img_grey, [distance], angles, normed=True)[:, :, 0, :]
                self.glcm = np.mean(self.glcm, axis=2)
                self.glcm = self.glcm[1:, 1:]

                # greylevel distance statistics
                self.glds = [np.sum(self.glcm.diagonal(n) + np.sum(self.glcm.diagonal(-n))) for n in range(256)]
                self.glds = self.glds / np.sum(self.glds)
                self.glds_diff = np.diff(self.glds)

            def __str__(self):
                print('Texture Analysis:\n')
                print(f'    mean: {self.get_mean()}')
                print(f'    standard deviation: {self.get_standard_deviation()}')
                print(f'    glcm contrast:{self.get_glcm_contrast()}')
                print(f'    glds skewness 0.25:{self.get_glds_skewness(0.25)}')
                print(f'    glds skewness 0.5:{self.get_glds_skewness(0.5)}')
                print(f'    glds skewness 0.75:{self.get_glds_skewness(0.75)}')

            def get_mean(self):
                return cv.mean(self.img, mask=self.mask)

            def get_standard_deviation(self):
                _, stddev = cv.meanStdDev(self.img, mask=self.mask)
                return stddev

            def get_glcm_contrast(self):
                i, j = np.indices(self.glcm.shape)
                coefficients = ((i - j) ** 2).astype(int)
                return np.sum(coefficients * self.glcm)

            def get_glds_skewness(self, proportion):
                level = 0
                for i, val in enumerate(self.glds):
                    level += val
                    if level >= proportion:
                        return i

    # TODO transparency
    # TODO edges
    # TODO interpretation

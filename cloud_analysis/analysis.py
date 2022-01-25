import os

import cv2 as cv
import numpy as np

from cloud_detection.cloud_filter import CloudFilter

path = os.path.realpath(__file__).removesuffix(r'cloud_analysis\analysis.py')

# for a more detailed explanation of the methods
# see http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth06.pdf
# and http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf


BORDER_DISTANCE = 972


class Analysis:

    # TODO convert to multiple functions
    def __init__(self, orig, num_clouds, border_threshold=0.1, border_width=25):
        self.orig = orig
        self.height, self.width, self.channels = self.orig.shape
        self.center = np.array([self.height / 2, self.width / 2])

        self.mask = self._get_mask()

        self.contours = self._get_contours(num_clouds, border_threshold=border_threshold, border_width=border_width)

        self.clouds = self._get_clouds()

    def __str__(self):
        return f'dimensions: {self.orig.shape}\nnumber of clouds: {len(self.clouds)}'

    def _get_mask(self):
        cloud_filter = CloudFilter()
        mask, _ = cloud_filter.evaluate_image(self.orig)
        return cv.resize(mask, (self.width, self.height))

    def _get_contours(self, num_clouds, border_threshold, border_width):
        all_contours, _ = cv.findContours(cv.medianBlur(self.mask, 3), cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
        all_contours = [np.squeeze(contour) for contour in all_contours]

        norm_contours = [np.linalg.norm(contour - self.center, axis=1) for contour in all_contours]

        border_ratios = []
        for i, contour in enumerate(norm_contours):
            num_border_pixels = np.count_nonzero([np.abs(distance - BORDER_DISTANCE) < border_width
                                                  or all_contours[i][n][0] < border_width
                                                  or all_contours[i][n][0] > self.height - border_width
                                                  for n, distance in enumerate(contour)])
            border_ratios.append(num_border_pixels / cv.arcLength(all_contours[i], True))
        print(border_ratios)
        non_border_contours = []
        for i, contour in enumerate(all_contours):
            if border_ratios[i] <= border_threshold:
                non_border_contours.append(contour)
        return non_border_contours
        # areas = np.array([cv.contourArea(contour) for contour in non_border_contours])
        # print(border_ratios)
        # print(max(areas))
        # max_areas = np.sort(areas)[-num_clouds:]
        # return [non_border_contours[np.where(areas == max_area)[0][0]] for max_area in max_areas]

    def _get_clouds(self):

        clouds = []
        for contour in self.contours:
            mask = np.zeros((self.height, self.width), np.uint8)
            cv.drawContours(mask, [contour], 0, (255, 255, 255), -1)
            img = cv.bitwise_and(self.orig, self.orig, mask=mask)
            clouds.append(self.Cloud(self.orig, img, mask, contour))
        return clouds

    class Cloud:
        def __init__(self, orig, img, mask, contour):
            self.orig = orig
            self.img = img
            self.mask = mask
            self.contour = contour

            self.shape = self.Shape(self.contour)
            self.texture = self.Texture(self.img, self.mask)

        def __str__(self):
            return f'dimensions: {self.img.shape}\n'

        class Shape:

            def __init__(self, contour):
                self.contour = contour
                self.contour_perimeter = cv.arcLength(self.contour, True)
                self.contour_area = cv.contourArea(self.contour)
                self.hull = cv.convexHull(self.contour)
                self.hull_perimeter = cv.arcLength(self.hull, True)
                self.hull_area = cv.contourArea(self.hull)

            def __str__(self):
                out = [f'Shape Analysis:',
                       f'   contour perimeter: {self.contour_perimeter}',
                       f'   hull perimeter: {self.hull_perimeter}',
                       f'   contour area: {self.contour_area}',
                       f'   hull area: {self.hull_area}',
                       f'   roundness: {self.roundness()}',
                       f'   rectangularity: {self.rectangularity()}',
                       f'   convexity: {self.convexity()}',
                       f'   compactness: {self.compactness()}',
                       f'   solidity: {self.solidity()}',
                       f'   elongation: {self.elongation()}', ]
                return '\n'.join(out)

            def roundness(self):
                return (4 * np.pi * self.contour_area) / (self.hull_perimeter ** 2)

            def rectangularity(self):
                _, (width, height), angle = cv.minAreaRect(self.contour)
                return self.contour_area / (width * height)

            def convexity(self):
                return self.hull_perimeter / self.contour_perimeter

            def compactness(self):
                return (4 * np.pi * self.contour_area) / (self.contour_perimeter ** 2)

            def solidity(self):
                return self.contour_area / self.hull_area

            def elongation(self):
                _, (width, height), angle = cv.minAreaRect(self.contour)
                return min(width, height) / max(width, height)

        class Texture:
            def __init__(self, img, mask):
                self.img = img
                self.mask = mask
                self.grey = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

            def __str__(self):
                out = ['Texture Analysis:',
                       f'   mean: {self.mean()}',
                       f'   standard deviation: {self.std()}',
                       f'   transparency: {self.transparency()}', ]
                return '\n'.join(out)

            def distribution(self):
                data = []
                for n in range(3):
                    channel = np.array(self.img[:, :, n].ravel())
                    data.append(channel[channel.nonzero()])
                return data

            def mean(self):
                return np.mean(self.img, axis=(0, 1), where=np.nonzero(self.mask))

            def std(self):
                _, std = cv.meanStdDev(self.img, mask=self.mask)
                std = (std[0][0], std[1][0], std[2][0])
                return std

            def transparency(self):
                sat = cv.cvtColor(self.img, cv.COLOR_RGB2HSV)[:, :, 0]
                inverse = np.where(sat == 0, sat, 255 - sat)
                return np.sum(inverse) / np.count_nonzero(inverse)

        def altitude(self):
            # https://en.wikipedia.org/wiki/Cloud_base#Measurement
            # https://en.wikipedia.org/wiki/Dew_point#Simple_approximation
            # altitude = surface temperature - dew point * 400 + surface level
            # dew point = temperature - (100 - humidity)/5 above 50% relative humidity
            # unknown: surface temperature and humidity - weather stations on earth?
            pass

        def edges(self, num_samples, in_steps, out_steps, regr_distance=1, regr_length=1):
            # num_pts = np.floor(self.contour.shape[0] * sample_proportion).astype('int')
            indices = np.sort(np.linspace(0, self.contour.shape[0] - 1 - regr_distance, num=num_samples).astype('int'))
            sample_points = self.contour[indices]

            regr_vectors = self.contour[indices + regr_distance] - sample_points
            regr_vectors = regr_vectors * regr_length / np.tile(np.linalg.norm(regr_vectors, axis=1), (2, 1)).T

            perpendicular_vectors = np.roll(regr_vectors, 1, axis=1)
            perpendicular_vectors[:, 0] *= -1

            spans = [[np.floor((vector * t) + sample_points[n]).astype('int')
                      for t in range(-in_steps, out_steps + 1)]
                     for n, vector in enumerate(perpendicular_vectors)]
            spans = np.array(spans).astype('int')

            edges = np.array([[self.orig[point[0], point[1]] for point in span] for span in spans])

            return np.mean(np.diff(np.mean(edges, axis=0), axis=0), axis=0)

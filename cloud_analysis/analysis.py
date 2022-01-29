import cv2 as cv
import numpy as np

import constants
from cloud_detection.cloud_filter import CloudFilter


class Analysis:

    def __init__(self, orig, num_clouds, border_threshold=0.1, border_width=5):
        if type(orig) is not np.ndarray:
            raise TypeError('orig should be of type ndarray')
        if len(orig.shape) != 3:
            raise TypeError('orig should have 3 channels')
        if orig.shape[2] != 3:
            raise TypeError('orig should be a RGB image')
        if type(num_clouds) is not int:
            raise TypeError('num_clouds should be of type int')
        if type(border_threshold) is not int:
            if type(border_threshold) is not float:
                raise TypeError(
                    'border_threshold should be of type int or float')
        if type(border_width) is not int:
            raise TypeError('border_width should be of type int')

        self.orig = orig
        self.height, self.width, self.channels = self.orig.shape
        self.center = np.array([self.height / 2, self.width / 2])

        self.mask = self._get_mask()
        self.contours = self._get_contours(num_clouds, border_threshold,
                                           border_width)
        self.clouds = self._get_clouds()

    def _get_mask(self):
        cloud_filter = CloudFilter()
        mask, _ = cloud_filter.evaluate_image(self.orig)
        if not np.any(mask):
            raise ValueError('orig has no clouds')

        return cv.resize(mask, (self.width, self.height))

    def _get_contours(self, num_clouds, border_threshold, border_width):

        # get a tuple of arrays of all contours in the image
        all_contours, _ = cv.findContours(cv.medianBlur(self.mask, 3),
                                          cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
        all_contours = [np.squeeze(contour) for contour in all_contours]
        if len(all_contours) < num_clouds:
            raise ValueError('orig has not enough clouds to return')

        # get the distance from the center for each point of the contours
        norm_contours = [np.linalg.norm(contour - self.center, axis=1)
                         for contour in all_contours]

        # get the ratio of pixels on the contours which lie on the edge of
        # the visible area to the respective contours
        border_ratios = []
        for i, contour in enumerate(norm_contours):
            num_border_pixels = np.count_nonzero(
                [np.abs(distance - constants.BORDER_DISTANCE) < border_width
                 or all_contours[i][n][0] < border_width
                 or all_contours[i][n][0] > self.height - border_width
                 for n, distance in enumerate(contour)])
            border_ratio = num_border_pixels / cv.arcLength(
                cv.convexHull(all_contours[i]), True)
            border_ratios.append(border_ratio)

        # get all contours whose border ratio does not exceed the threshold
        non_border_contours = [contour for i, contour in
                               enumerate(all_contours)
                               if border_ratios[i] <= border_threshold]
        if len(non_border_contours) < num_clouds:
            raise ValueError('orig has not enough clouds which are inside the '
                             'visible area to return')

        # take the largest contours
        areas = np.array(
            [cv.contourArea(contour) for contour in non_border_contours])
        max_areas = np.sort(areas)[-num_clouds:]
        largest_contours = [
            non_border_contours[np.where(areas == max_area)[0][0]]
            for max_area in max_areas]

        return largest_contours

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
            self.height, self.width, self.channels = self.orig.shape
            self.img = img
            self.mask = mask
            self.contour = contour

            self.shape = self.Shape(self.contour)
            self.texture = self.Texture(self.img, self.mask)

        class Shape:

            def __init__(self, contour):
                self.contour = contour
                self.contour_perimeter = cv.arcLength(self.contour, True)
                self.contour_area = cv.contourArea(self.contour)
                self.hull = cv.convexHull(self.contour)
                self.hull_perimeter = cv.arcLength(self.hull, True)
                self.hull_area = cv.contourArea(self.hull)

            def roundness(self):
                return (4 * np.pi * self.contour_area) / (
                        self.hull_perimeter ** 2)

            def convexity(self):
                return self.hull_perimeter / self.contour_perimeter

            def compactness(self):
                return (4 * np.pi * self.contour_area) / (
                        self.contour_perimeter ** 2)

            def solidity(self):
                return self.contour_area / self.hull_area

            def rectangularity(self):
                _, (width, height), angle = cv.minAreaRect(self.contour)
                return self.contour_area / (width * height)

            def elongation(self):
                _, (width, height), angle = cv.minAreaRect(self.contour)
                return min(width, height) / max(width, height)

        class Texture:
            def __init__(self, img, mask):
                self.img = img
                self.mask = mask
                self.grey = cv.cvtColor(self.img, cv.COLOR_RGB2GRAY)

            def distribution(self):
                data = []
                for n in range(3):
                    channel = np.array(self.img[:, :, n].ravel())
                    non_zero = channel[channel.nonzero()]
                    data.append(non_zero)
                return np.array(data)

            def mean(self):
                return cv.mean(self.img, mask=self.mask)

            def std(self):
                _, std = cv.meanStdDev(self.img, mask=self.mask)
                std = (std[0][0], std[1][0], std[2][0])
                return std

            def transparency(self):
                sat = cv.cvtColor(self.img, cv.COLOR_RGB2HSV)[:, :, 0]
                inverse = np.where(sat == 0, sat, 255 - sat)
                if np.count_nonzero(inverse) != 0:
                    out = np.sum(inverse) / np.count_nonzero(inverse)
                else:
                    out = 0
                return out

        def mean_diff_edges(self, num_samples, in_steps, out_steps,
                            regr_dist=3, regr_len=1):
            if type(num_samples) is not int:
                raise TypeError('num_samples should be of type int')
            if type(in_steps) is not int:
                raise TypeError('in_steps should be of type int')
            if type(out_steps) is not int:
                raise TypeError('out_steps should be of type int')
            if type(regr_dist) is not int:
                raise TypeError('regr_dist should be of type int')
            if type(regr_len) is not int:
                if type(regr_len) is not float:
                    raise TypeError('regr_len should be of type int or float')

            regr_vec = np.roll(self.contour, regr_dist, axis=0)
            regr_vec -= self.contour
            regr_vec_norm = np.linalg.norm(regr_vec, axis=1)
            regr_vec = regr_vec * regr_len / np.tile(regr_vec_norm, (2, 1)).T

            perp_vec = np.roll(regr_vec, 1, axis=1)
            perp_vec[:, 0] *= -1

            span_range = np.arange(-in_steps, out_steps + 1)[:, np.newaxis]
            spans = [np.matmul(span_range, vec[np.newaxis]) + self.contour[n]
                     for n, vec in enumerate(perp_vec)]
            spans = np.floor(np.array(spans)).astype('int')

            def is_in_image(span):
                return (np.all(self.height > span[:, 0])
                        and np.all(span[:, 0] >= 0)
                        and np.all(self.width > span[:, 1])
                        and np.all(span[:, 1] >= 0))

            valid_spans = [span for span in spans if is_in_image(span)]
            valid_spans = np.array(valid_spans)

            idx = np.linspace(0, valid_spans.shape[0] - 1, num=num_samples)
            sample_spans = valid_spans[idx.astype('int')]

            edges = np.array([[self.orig[point[0], point[1]]
                               for point in span]
                              for span in sample_spans])

            out = np.mean(np.diff(np.mean(edges, axis=0), axis=0), axis=0)

            return out

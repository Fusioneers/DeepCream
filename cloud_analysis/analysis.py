"""Module containing a class to analyse pictures taken with the AstroPi.

This module contains a single class named Analysis, which is initialised with a
given input image and creates a number of objects i.e. clouds from which
properties such as convexity or transparency can be read off.

    Typical usage example:

    import cv2 as cv

    image = cv.imread(path)
    number_of_clouds = 5
    analysis = Analysis(image, number_of_clouds)

    print(f'Convexity: {analysis.clouds[0].convexity}')
    print(f'Transparency: {analysis.clouds[0].transparency}')
    cv.imshow(analysis.clouds[0].img)
"""

import cv2 as cv
import numpy as np

from cloud_detection.cloud_filter import CloudFilter
# TODO save only the proportion, get const by image size
from constants import BORDER_DISTANCE, BORDER_WIDTH


class Analysis:
    """A class for analysing clouds in an image taken with the AstroPi.

        This class creates a number of clouds from an image given at its
        initialisation. Those are objects containing multiple attributes and
        methods which return information about the nature of the cloud.

        Attributes:
            orig:
                The original image which is passed during initialisation.
            height:
                The height in pixels of orig.
            width:
                The width in pixels of orig.
            mask:
                A numpy array which corresponds to an image which has the same
                height and width as orig, with a 255 for a pixel which is
                estimated to be a cloud and a 0 otherwise. It has the same
                shape as orig, but only a single channel.
            contours:
                A tuple containing numpy arrays which represent the contours
                of detected clouds. Only clouds whose contour does not overlap
                too much with the edge of the visible area are valid. The
                largest clouds are returned.
            clouds:
                A list of cloud objects resembling each a cloud in the image.
    """

    def __init__(self, orig: np.ndarray, num_clouds: int,
                 border_threshold: float = 0.1):
        """Initialises Analysis.

        Args:
            orig:
                The original image which is passed during initialisation.
            num_clouds:
                The number of clouds which are created.
            border_threshold:
                The allowed proportion of the number of pixels which
                lie on the edge of the visible area in the convex hull of a
                cloud. A low value means that clouds have to be fully inside
                the visible area to be recognised.
        """

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

        self.orig = orig
        self.height, self.width, _ = self.orig.shape

        self.mask = self._get_mask()
        self.contours = self._get_contours(num_clouds, border_threshold)
        self.clouds = self._get_clouds()

    def _get_mask(self) -> np.ndarray:
        """Gets the cloud mask from CloudFilter."""
        cloud_filter = CloudFilter()
        mask, _ = cloud_filter.evaluate_image(self.orig)
        if not np.any(mask):
            raise ValueError('orig has no clouds')

        return cv.resize(mask, (self.width, self.height))

    def _get_contours(self, num_clouds: int,
                      border_threshold: float) -> np.ndarray:
        """Gets the contours of the clouds.

        TODO function description


        Args:
            num_clouds:
                The number of clouds which are created.
            border_threshold:
                The allowed proportion of the number of pixels which
                lie on the edge of the visible area in the convex hull of a
                cloud. A low value means that clouds have to be fully inside
                the visible area to be recognised.

        Returns:
            A tuple containing numpy arrays which represent the coordinates of
            the contours.
        """

        # TODO border_threshold 0 for no validation
        all_contours, _ = cv.findContours(cv.medianBlur(self.mask, 3),
                                          cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
        all_contours = [np.squeeze(contour) for contour in all_contours]
        if len(all_contours) < num_clouds:
            raise ValueError('orig has not enough clouds to return')

        center = np.array([self.height / 2, self.width / 2])
        norm_contours = [np.linalg.norm(contour - center, axis=1)
                         for contour in all_contours]

        border_ratios = []
        for i, contour in enumerate(norm_contours):
            num_border_pixels = np.count_nonzero(
                [np.abs(distance - BORDER_DISTANCE) < BORDER_WIDTH
                 or all_contours[i][n][0] < BORDER_WIDTH
                 or all_contours[i][n][0] > self.height - BORDER_WIDTH
                 for n, distance in enumerate(contour)])
            border_ratio = num_border_pixels / cv.arcLength(
                cv.convexHull(all_contours[i]), True)
            border_ratios.append(border_ratio)

        non_border_contours = [contour for i, contour in
                               enumerate(all_contours)
                               if border_ratios[i] <= border_threshold]
        if len(non_border_contours) < num_clouds:
            raise ValueError('orig has not enough clouds which are inside the '
                             'visible area to return')

        areas = np.array(
            [cv.contourArea(contour) for contour in non_border_contours])
        max_areas = np.sort(areas)[-num_clouds:]
        largest_contours = [
            non_border_contours[np.where(areas == max_area)[0][0]]
            for max_area in max_areas]

        return largest_contours

    def _get_clouds(self) -> list:
        """Creates a list of the clouds."""

        clouds = []
        for contour in self.contours:
            mask = np.zeros((self.height, self.width), np.uint8)
            cv.drawContours(mask, [contour], 0, (255, 255, 255), -1)
            img = cv.bitwise_and(self.orig, self.orig, mask=mask)
            clouds.append(self.Cloud(self.orig, img, mask, contour))
        return clouds

    class Cloud:
        def __init__(self, orig: np.ndarray, img: np.ndarray, mask: np.ndarray,
                     contour: tuple):
            self.orig = orig
            self.height, self.width, self.channels = self.orig.shape
            self.img = img
            self.mask = mask
            self.contour = contour

            self.contour_perimeter = cv.arcLength(self.contour, True)
            self.contour_area = cv.contourArea(self.contour)
            self.hull = cv.convexHull(self.contour)
            self.hull_perimeter = cv.arcLength(self.hull, True)
            self.hull_area = cv.contourArea(self.hull)

        def roundness(self) -> float:
            return (4 * np.pi * self.contour_area) / (
                    self.hull_perimeter ** 2)

        def convexity(self) -> float:
            return self.hull_perimeter / self.contour_perimeter

        def compactness(self) -> float:
            return (4 * np.pi * self.contour_area) / (
                    self.contour_perimeter ** 2)

        def solidity(self) -> float:
            return self.contour_area / self.hull_area

        def rectangularity(self) -> float:
            _, (width, height), angle = cv.minAreaRect(self.contour)
            return self.contour_area / (width * height)

        def elongation(self) -> float:
            _, (width, height), angle = cv.minAreaRect(self.contour)
            return min(width, height) / max(width, height)

        def distribution(self) -> list:
            data = []
            for n in range(3):
                channel = np.array(self.img[:, :, n].ravel())
                non_zero = channel[channel.nonzero()]
                data.append(non_zero)
            return np.array(data)

        def mean(self) -> list:
            return cv.mean(self.img, mask=self.mask)

        def std(self) -> list:
            _, std = cv.meanStdDev(self.img, mask=self.mask)
            std = (std[0][0], std[1][0], std[2][0])
            return std

        def transparency(self) -> float:
            sat = cv.cvtColor(self.img, cv.COLOR_RGB2HSV)[:, :, 0]
            inverse = np.where(sat == 0, sat, 255 - sat)
            if np.count_nonzero(inverse) != 0:
                out = np.sum(inverse) / np.count_nonzero(inverse)
            else:
                out = 0
            return out

        def mean_diff_edges(self, num_samples: int, in_steps: int,
                            out_steps: int, regr_dist: int = 3,
                            regr_len: float = 1) -> float:
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

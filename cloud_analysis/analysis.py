import cv2 as cv
import numpy as np

from cloud_detection.cloud_filter import CloudFilter

# for a more detailed explanation of the methods
# see http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth06.pdf
# and http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf

BORDER_DISTANCE = 972


def check_type(var, val, msg: str):
    if var != val:
        raise TypeError(msg)


def check_value(val, msg: str):
    if not val:
        raise ValueError(msg)


class Analysis:

    def __init__(self, orig: np.ndarray, num_clouds: int, border_threshold: float = 0.1, border_width: int = 5):
        check_type(orig, np.ndarray, 'The image should be a numpy array')
        check_type(len(orig.shape), 3, 'The image has to be an array of the shape (height, width, channels)')
        check_type(orig.shape[2], 3, 'The image has to be a RGB image')
        check_type(num_clouds, int, 'The number of clouds has to be an integer')
        check_type(border_threshold, float, 'The threshold for the border proportion of the clouds has to be a number')
        check_type(border_width, int, 'The width of the border has to be an integer (in pixel)')

        self.orig = orig
        self.height, self.width, self.channels = self.orig.shape
        self.center = np.array([self.height / 2, self.width / 2])

        self.mask = self._get_mask()
        self.contours = self._get_contours(num_clouds, border_threshold, border_width)
        self.clouds = self._get_clouds()

    def _get_mask(self) -> np.ndarray:
        cloud_filter = CloudFilter()
        mask, _ = cloud_filter.evaluate_image(self.orig)
        check_value(np.any(mask), 'the image has no clouds')

        return cv.resize(mask, (self.width, self.height))

    def _get_contours(self, num_clouds: int, border_threshold: float, border_width: int) -> np.ndarray:

        # get a tuple of arrays of all contours in the image
        all_contours, _ = cv.findContours(cv.medianBlur(self.mask, 3), cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
        all_contours = [np.squeeze(contour) for contour in all_contours]
        check_value(len(all_contours) < num_clouds, 'The image has not enough clouds to return')

        # get the distance from the center for each point of the contours
        norm_contours = [np.linalg.norm(contour - self.center, axis=1) for contour in all_contours]

        # get the ratio of pixels on the contours which lie on the edge of the visible area to the respective contours
        border_ratios = []
        for i, contour in enumerate(norm_contours):
            num_border_pixels = np.count_nonzero([np.abs(distance - BORDER_DISTANCE) < border_width
                                                  or all_contours[i][n][0] < border_width
                                                  or all_contours[i][n][0] > self.height - border_width
                                                  for n, distance in enumerate(contour)])
            border_ratio = num_border_pixels / cv.arcLength(cv.convexHull(all_contours[i]), True)
            border_ratios.append(border_ratio)

        # get all contours whose border ratio does not exceed a specific threshold
        non_border_contours = [contour for i, contour in enumerate(all_contours)
                               if border_ratios[i] <= border_threshold]
        check_value(len(non_border_contours) < num_clouds, 'The image has not enough clouds'
                                                           'which are inside the visible area to return')

        # take the largest contours
        areas = np.array([cv.contourArea(contour) for contour in non_border_contours])
        max_areas = np.sort(areas)[-num_clouds:]
        largest_contours = [non_border_contours[np.where(areas == max_area)[0][0]] for max_area in max_areas]

        return largest_contours

    def _get_clouds(self) -> list:
        clouds = []
        for contour in self.contours:
            mask = np.zeros((self.height, self.width), np.uint8)
            cv.drawContours(mask, [contour], 0, (255, 255, 255), -1)
            img = cv.bitwise_and(self.orig, self.orig, mask=mask)
            clouds.append(self.Cloud(self.orig, img, mask, contour))
        return clouds

    class Cloud:
        def __init__(self, orig: np.ndarray, img: np.ndarray, mask: np.ndarray, contour: tuple):
            self.orig = orig
            self.height, self.width, self.channels = self.orig.shape
            self.img = img
            self.mask = mask
            self.contour = contour

            self.shape = self.Shape(self.contour)
            self.texture = self.Texture(self.img, self.mask)

        class Shape:

            def __init__(self, contour: tuple):
                self.contour = contour
                self.contour_perimeter = cv.arcLength(self.contour, True)
                self.contour_area = cv.contourArea(self.contour)
                self.hull = cv.convexHull(self.contour)
                self.hull_perimeter = cv.arcLength(self.hull, True)
                self.hull_area = cv.contourArea(self.hull)

            def roundness(self) -> float:
                return (4 * np.pi * self.contour_area) / (self.hull_perimeter ** 2)

            def convexity(self) -> float:
                return self.hull_perimeter / self.contour_perimeter

            def compactness(self) -> float:
                return (4 * np.pi * self.contour_area) / (self.contour_perimeter ** 2)

            def solidity(self) -> float:
                return self.contour_area / self.hull_area

            def rectangularity(self) -> float:
                _, (width, height), angle = cv.minAreaRect(self.contour)
                return self.contour_area / (width * height)

            def elongation(self) -> float:
                _, (width, height), angle = cv.minAreaRect(self.contour)
                return min(width, height) / max(width, height)

        class Texture:
            def __init__(self, img: np.ndarray, mask: np.ndarray):
                self.img = img
                self.mask = mask
                self.grey = cv.cvtColor(self.img, cv.COLOR_RGB2GRAY)

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

        def altitude(self):
            # https://en.wikipedia.org/wiki/Cloud_base#Measurement
            # https://en.wikipedia.org/wiki/Dew_point#Simple_approximation
            # altitude = surface temperature - dew point * 400 + surface level
            # dew point = temperature - (100 - humidity)/5 above 50% relative humidity
            # unknown: surface temperature and humidity - weather stations on earth?
            pass

        def mean_diff_edges(self, num_samples: int, in_steps: int, out_steps: int,
                            regr_dist: int = 3, regr_len: float = 1) -> float:
            check_type(num_samples, int, 'num_samples should be an integer')
            check_type(in_steps, int, 'in_steps should be an integer')
            check_type(out_steps, int, 'out_steps should be an integer')
            check_type(regr_dist, int, 'regr_dist should be an integer')
            check_type(regr_len, float, 'regr_length')

            regr_vectors = np.roll(self.contour, regr_dist, axis=0) - self.contour
            regr_vectors = regr_vectors * regr_len / np.tile(np.linalg.norm(regr_vectors, axis=1), (2, 1)).T

            perp_vectors = np.roll(regr_vectors, 1, axis=1)
            perp_vectors[:, 0] *= -1

            spans = np.floor(np.array([np.matmul(np.arange(-in_steps, out_steps + 1)[:, np.newaxis], vector[np.newaxis])
                                       + self.contour[n] for n, vector in enumerate(perp_vectors)])).astype('int')

            valid_spans = np.array([span for span in spans if
                                    np.all(np.logical_and(np.logical_and(self.height > span[:, 0], span[:, 0] >= 0),
                                                          np.logical_and(self.width > span[:, 1], span[:, 1] >= 0)))])

            sample_spans = valid_spans[np.linspace(0, valid_spans.shape[0] - 1, num=num_samples).astype('int')]

            edges = np.array([[self.orig[point[0], point[1]] for point in span] for span in sample_spans])

            out = np.mean(np.diff(np.mean(edges, axis=0), axis=0), axis=0)

            return out

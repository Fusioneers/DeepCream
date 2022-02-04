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

import logging

logging.info('Started DeepCream/cloud_analysis/analysis.py')
import cv2 as cv
import numpy as np

from DeepCream.cloud_detection.cloud_filter import CloudFilter
from DeepCream.constants import default_step_len, default_appr_dist


# TODO logging
# TODO update documentation
# TODO add more exception clauses and raises


class Analysis:
    """A class for analysing clouds in an image taken with the AstroPi.

        This class creates a number of clouds from an image given at its
        initialisation. These are objects containing multiple attributes and
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

    def __init__(self, orig: np.ndarray,
                 min_size_proportion: float,
                 border_width: int,
                 contrast_threshold: float):
        """Initialises Analysis.

        Args:
            orig:
                The original RGB image which is passed during initialisation.
                It should have the shape (height, width, channels).
            num_clouds:
                The number of clouds which are created.
            border_threshold:
                The allowed proportion of the number of pixels which
                lie on the edge of the visible area in the convex hull of a
                cloud. A low value means that clouds have to be fully inside
                the visible area to be recognised.
        """

        self.orig = orig
        self.height, self.width, _ = self.orig.shape

        self.mask = self._get_mask()
        logging.info('Created mask')

        self.contours = self._get_contours()
        logging.info('Created contours')

        self.clouds = self._get_clouds(self.contours, min_size_proportion,
                                       border_width, contrast_threshold)
        logging.info('Created clouds')

    def _get_mask(self) -> np.ndarray:
        """Gets the cloud mask from CloudFilter.

        Returns:
            A numpy array which corresponds to an image which has the same
            height and width as orig, with a 255 for a pixel which is
            estimated to be a cloud and a 0 otherwise. It has the same shape
            as orig, but only a single channel i.e. shape (height, width).

        Raises:
            ValueError: Orig has no clouds.
        """
        cloud_filter = CloudFilter()
        logging.debug('Initialised CloudFilter')

        mask, _ = cloud_filter.evaluate_image(self.orig)
        logging.debug('Evaluated orig with CloudFilter')

        if not np.any(mask):
            logging.warning('Orig has no clouds')
            raise ValueError('Orig has no clouds')

        out = cv.resize(mask, (self.width, self.height))
        logging.debug('Resized mask')

        return out

    def _get_contours(self) -> tuple[np.ndarray]:
        """Gets the contours of the clouds.

        This function computes the contours of orig. those get filtered by the
        proportion of pixels they share with the edge of the visible area to
        ensure that only whole clouds get analysed. The num_clouds largest
        clouds (sorted by area) are then returned.

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

        Raises:
            ValueError: Orig has not enough clouds to return.
            ValueError: Orig has not enough clouds which are inside the visible
                area to return.
        """

        contours, _ = cv.findContours(cv.medianBlur(self.mask, 3),
                                      cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
        logging.debug('Found contours with cv.findContours')

        contours = [np.squeeze(contour) for contour in contours]
        logging.debug('Reformatted contours')

        return contours

    def _get_clouds(self,
                    contours: list,
                    min_size_proportion: float,
                    border_width: int,
                    contrast_threshold: float) -> list:

        min_size = min_size_proportion * self.height * self.width

        clouds = []
        for contour in contours:
            mask = np.zeros((self.height, self.width), np.uint8)
            cv.drawContours(mask, [contour], 0, (255, 255, 255), -1)
            img = cv.bitwise_and(self.orig, self.orig, mask=mask)
            clouds.append(self.Cloud(self.orig, img, mask, contour))
        logging.debug('Created list of all clouds')

        big_clouds = list(
            filter(lambda cloud: cloud.contour_area >= min_size, clouds))
        logging.debug('Filtered clouds by size')

        non_image_border_clouds = list(
            filter(lambda cloud: np.all(
                cloud.contour[:, 1] >= border_width) and np.all(
                cloud.contour[:, 1] <= cloud.height - border_width),
                   big_clouds))
        logging.debug('Filtered clouds by image border')

        def check_valid(cloud):
            try:
                max_contrast = np.max(
                    cloud.diff_edges(border_width, border_width))
                out = max_contrast <= contrast_threshold
            except ValueError as err:
                logging.warning(f'ValueError occurred at check_valid: {err}')
                out = False
            return out

        visible_area_clouds = list(
            filter(check_valid, non_image_border_clouds))
        logging.debug('Filtered clouds by visible area border')

        return visible_area_clouds

    class Cloud:
        """A single cloud in orig.

        This class corresponds to a single detected cloud. It has multiple
        fields and methods to gain information about the cloud.

        Attributes:
            orig:
                The original image which is passed during initialisation.
            height:
                The height in pixels of orig.
            width:
                The width in pixels of orig.
            img:
                A mostly black image with only the cloud itself visible. It
                can also be obtained by combining orig with mask.
            mask:
                A numpy array which corresponds to an image which has the same
                height and width as orig, with a 255 for a pixel which belongs
                cloud and a 0 otherwise. It has the same shape as orig, but
                only a single channel.
            contour:
                A numpy array containing the coordinates of the edge of the
                cloud.
            contour_perimeter:
                The perimeter of the contour.
            contour_area:
                The area of the contour.
            hull:
                a numpy array similar to contour. It represents the convex
                hull around contour.
            hull_perimeter:
                The perimeter of the hull.
            hull_area:
                The area of the hull.
        """

        def __init__(self, orig: np.ndarray,
                     img: np.ndarray,
                     mask: np.ndarray,
                     contour: np.ndarray):
            """Initialises the cloud.

            Args:
                orig:
                    The original RGB image which is passed during
                    initialisation. It should have the shape
                    (height, width, channels).
                img:
                    A mostly black image with only the cloud itself visible.
                    It can also be obtained by combining orig with mask.
                mask:
                    A numpy array which corresponds to an image which has the
                    same height and width as orig, with a 255 for a pixel which
                    belongs cloud and a 0 otherwise. It has the same shape as
                    orig, but only a single channel.
                contour:
                    A numpy array containing the coordinates of the edge of the
                    cloud.
            """

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

            logging.debug('Initialised cloud')

        def roundness(self) -> float:
            """Gets the roundness of the cloud."""
            return (4 * np.pi * self.contour_area) / (
                    self.hull_perimeter ** 2)

        def convexity(self) -> float:
            """Gets the convexity of the cloud."""
            return self.hull_perimeter / self.contour_perimeter

        def compactness(self) -> float:
            """Gets the compactness of the cloud."""
            return (4 * np.pi * self.contour_area) / (
                    self.contour_perimeter ** 2)

        def solidity(self) -> float:
            """Gets the solidity of the cloud."""
            return self.contour_area / self.hull_area

        def rectangularity(self) -> float:
            """Gets the rectangularity of the cloud."""
            _, (width, height), angle = cv.minAreaRect(self.contour)
            return self.contour_area / (width * height)

        def elongation(self) -> float:
            """Gets the elongation of the cloud."""
            _, (width, height), angle = cv.minAreaRect(self.contour)
            return min(width, height) / max(width, height)

        def mean(self) -> list:
            """Gets the mean of each channel inside the cloud"""
            return cv.mean(self.img, mask=self.mask)

        def std(self) -> list:
            """Gets the standard deviation of each channel inside the cloud."""
            _, std = cv.meanStdDev(self.img, mask=self.mask)
            std = (std[0][0], std[1][0], std[2][0])
            return std

        def transparency(self) -> float:
            """Gets the transparency of the cloud.

            The transparency is computed by averaging the deviation of each
            pixel in the cloud from pure white. This is less effective for
            light ground such as snow or if the cloud itself is colorful e.g.
            at sunset. Note that it is based on the saturation of an HSV image
            of the cloud.

            Returns:
                A value between 0 and 255. 255 means that the cloud is perfect
                grey, while 0 means a very colorful cloud.
            """

            sat = cv.cvtColor(self.img, cv.COLOR_RGB2HSV)[:, :, 0]
            inverse = np.where(sat == 0, sat, 255 - sat)
            if np.count_nonzero(inverse) != 0:
                out = np.sum(inverse) / np.count_nonzero(inverse)
            else:
                out = 0
            return out

        def edges(self,
                  in_steps: int,
                  out_steps: int,
                  appr_dist: int = default_appr_dist,
                  step_len: float = default_step_len) -> np.ndarray:
            """Gets samples of the surroundings of the contour of the cloud.

            First, perpendicular vectors to the contour are created. Those are
            estimated by computing the tangents for each pixel on the contour.
            Because pixels are discrete, they are actually secants from the
            pixel to the appr_dist-th one next to it. From those vectors the
            perpendicular vectors with length step_len are gained. For each
            perpendicular vector the span is computed i.e. an array of points
            which are a multiple of the original vector. Note that in_steps is
            the number of steps the span reaches into cloud from the boundary
            and out_steps the number of steps out of the cloud. These steps
            have the length step_len. Then, samples of the spans, which lie
            fully inside the visible area are returned.

            # TODO usage example

            Args:
                num_samples:
                    The number of samples of the edge to be returned.
                in_steps:
                    The number of steps the span reaches into cloud from the
                    boundary excluding the boundary itself.
                out_steps:
                    The number of steps the span reaches out of the cloud from
                    the boundary excluding the boundary itself.
                appr_dist:
                    The contour approximation vectors i.e. tangents of the
                    contour are estimated by a secant from the point itself to
                    the appr_dist next point. A high value of e.g. 10 means
                    that the approximation may be more accurate, while for a
                    very rough boundary it is better to choose a lower value.
                    A value smaller than 1 should be avoided.
                step_len:
                    The distance between each point in the span. A low value
                    gives a more dense overview of the edge, while a higher one
                    yields a wider range. Note that for a value of 1 some
                    points can be the same. A value smaller than 1 should be
                    avoided.

            Returns:
                A numpy array of shape
                (num_samples, in_steps + out_steps + 1, 3), which is a folded
                representative sample of the edge of the cloud.
            """

            appr_vec = np.roll(self.contour, appr_dist, axis=0)
            appr_vec -= self.contour
            appr_vec_norm = np.linalg.norm(appr_vec, axis=1)
            appr_vec = appr_vec * step_len / np.tile(appr_vec_norm, (2, 1)).T

            logging.debug('Created appr_vec')

            perp_vec = np.roll(appr_vec, 1, axis=1)
            perp_vec[:, 0] *= -1
            logging.debug('Created perp_vec')

            span_range = np.arange(-in_steps, out_steps + 1)[:, np.newaxis]
            spans = [np.matmul(span_range, vec[np.newaxis]) + self.contour[n]
                     for n, vec in enumerate(perp_vec)]
            spans = np.floor(np.array(spans)).astype('int')
            logging.debug('Created spans')

            valid_spans = [span for span in spans if
                           np.all(span[:, 0] > 0)
                           and np.all(span[:, 0] < self.height)
                           and np.all(span[:, 1] > 0)
                           and np.all(span[:, 1] < self.width)]
            valid_spans = np.array(valid_spans)
            logging.debug('Created valid_spans')

            if not valid_spans.size:
                logging.warning('The cloud has no valid spans')
                raise ValueError('The cloud has no valid spans')

            edges = np.array([[self.orig[point[0], point[1]]
                               for point in span]
                              for span in valid_spans])
            logging.debug('Created edges')

            return edges

        def diff_edges(self,
                       in_dist: float,
                       out_dist: float,
                       appr_dist: int = default_appr_dist,
                       step_len: float = default_step_len) -> np.ndarray:
            """A measurement for the roughness of the edge of the cloud.

            The function computes an average roughness of the edge of the cloud
            by averaging the absolute difference of the pixels near the
            boundary in each channel. For a more detailed description of the
            functionality and the arguments see Cloud.edges.

            Returns:
                A numpy array with the average change from one pixel to another
                in the direction of the boundary. It has shape 3.
            """
            edges = self.edges(int(np.floor(in_dist / step_len)),
                               int(np.floor(out_dist / step_len)),
                               appr_dist, step_len)
            return np.abs(np.diff(np.mean(edges, axis=0), axis=0)) / step_len

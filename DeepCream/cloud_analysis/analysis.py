"""Module containing a class to analyse pictures taken with the AstroPi.

This module contains a single class named Analysis, which is initialised with a
given input image and creates a number of objects i.e. clouds from which
properties such as convexity or transparency can be read off.

    Typical usage example:

    import cv2 as cv

    image = cv.imread(path)

    cloud_filter = CloudFilter()
    mask, _ = cloud_filter.evaluate_image(orig)
    mask = cv.resize(mask, (image.shape[1], image.shape[0]))

    max_number_of_clouds = 10
    analysis = Analysis(image, mask, max_number_of_clouds)

    print(f'Convexity: {analysis.clouds[0].convexity()}')
    print(f'Transparency: {analysis.clouds[0].transparency()}')
    cv.imshow(analysis.clouds[0].img)
"""

import logging

import cv2 as cv
import numpy as np

from DeepCream.constants import (DEFAULT_STEP_LEN,
                                 DEFAULT_APPR_DIST,
                                 DEFAULT_BORDER_WIDTH,
                                 DEFAULT_VAL_THRESHOLD)

logging.info('Started DeepCream/cloud_analysis/analysis.py')


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
            of the detected clouds. Only clouds whose contour does not
            overlap too much with the edge of the visible area are valid.
            Then the largest clouds are selected. If there are less valid
            clouds in the image than the specified value all valid clouds
            are returned.

            clouds:
            A list of cloud objects resembling each a cloud in the image.
    """

    def __init__(self,
                 orig: np.ndarray,
                 mask: np.ndarray,
                 max_num_clouds: int,
                 max_border_proportion: float,
                 border_width: int = DEFAULT_BORDER_WIDTH,
                 val_threshold: float = DEFAULT_VAL_THRESHOLD):
        """Initialises Analysis.

        Args:
            orig:
            The original RGB image which is passed during initialisation.
            It should have the shape (height, width, channels).

            mask:
            A mask which determines whether a pixel belongs to a clouds.
            It is a numpy array which corresponds to an image which has the
            same height and width as orig, with a 255 for a pixel which is
            estimated to be a cloud and a 0 otherwise. It should have the
            same shape as orig, but only a single channel i.e. shape
            (height, width).

            max_num_clouds:
            The maximum number of clouds to be created. If there
            are fewer clouds than the specified value in the image all
            valid clouds are returned.

            max_border_proportion:
            The maximum proportion of the convex hull a cloud is allowed to
            share with the edge of the visible area.

            border_width:
            The distance in pixels from the edge of the visible area to the
            contour a cloud is allowed to be.

            val_threshold:
            The value (as in HSV) that is considered to lie outside the
            visible area.
            """

        self.orig = orig
        self.height, self.width, _ = self.orig.shape

        self.mask = mask
        logging.info('Created mask')
        if not np.any(self.mask):
            self.contours = ()
            self.clouds = []
            logging.warning('Orig has no clouds')
        else:
            self.contours = self.__get_contours()
            logging.info('Created contours')

            self.clouds = self.__get_clouds(self.contours,
                                            max_num_clouds,
                                            max_border_proportion,
                                            border_width,
                                            val_threshold)
            logging.info(f'Created {len(self.clouds)} clouds')

    def __get_contours(self):
        """Gets the contours of the clouds.

        Returns:
        A list containing numpy arrays which represent the coordinates of
        the contours.
        """

        contours, _ = cv.findContours(cv.medianBlur(self.mask, 3),
                                      cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
        logging.debug('Found contours with cv.findContours')

        contours = [np.squeeze(contour) for contour in contours]
        logging.debug('Reformatted contours')

        return contours

    # TODO Test this function
    def __get_clouds(self,
                     contours: list,
                     max_num_clouds: int,
                     max_border_proportion: float,
                     border_width: float,
                     val_threshold: float) -> list:
        """Creates and filters the cloud objects.

        This method packs the contour, image, mask and orig of a cloud into a
        cloud object, filters and returns the largest of them. The filtering is
        based around the problem, that clouds, which share a major part of
        their border with the edge of the visible area, distort the final
        analysis because of their seemingly super smooth border. So clouds,
        which are near this edge, are called invalid and therefore not
        considered for further analysis.

        Args:
            contours:
            A list containing numpy arrays which represent the coordinates of
            the contours.

            max_num_clouds:
            The maximum number of clouds which should be returned. In most
            cases this value is satisfied, except for the images with very few
            clouds.

            max_border_proportion:
            The maximum proportion of the convex hull a cloud is allowed to
            share with the edge of the visible area.

            border_width:
            At the edge of the visible area there is a thin strip of pixels
            which neither belong to the outside i.e. clouds and water nor to
            the black surroundings. border_width is the distance in pixels from
            the certain-black to the certain-outside. A higher value means that
            the cloud has to be further away from the border to be considered
            valid.
        """

        all_clouds = []
        for contour in contours:
            mask = np.zeros((self.height, self.width), np.uint8)
            cv.drawContours(mask, [contour], 0, (255, 255, 255), -1)
            img = cv.bitwise_and(self.orig, self.orig, mask=mask)
            all_clouds.append(self.Cloud(self.orig, img, mask, contour))
        logging.debug('Created list of all clouds')

        # TODO test this
        def check_valid(cloud):
            edges = cloud.edges(border_width, border_width, convex_hull=True)
            if not edges.size:
                out = False
            else:
                min_vals = np.min(np.max(edges, axis=2), axis=1)
                border_proportion = np.count_nonzero(
                    min_vals >= val_threshold) / len(cloud.hull)
                out = border_proportion <= max_border_proportion
            return out

        if max_border_proportion == 1:
            if len(all_clouds) <= max_num_clouds:
                clouds = all_clouds
                logging.debug(
                    'There are less or equal valid clouds than max_num_clouds')
            else:
                clouds = all_clouds[:max_num_clouds]
                logging.debug('Filtered clouds by size')
        else:
            all_clouds = sorted(all_clouds,
                                key=lambda cloud:
                                getattr(cloud, 'contour_area'), reverse=True)

            # TODO test this
            clouds = []
            for cloud in all_clouds:
                if len(clouds) >= max_num_clouds:
                    break
                if np.all(cloud.hull[:, 1] >= border_width):
                    if np.all(cloud.hull[:, 1] <= cloud.height - border_width):
                        if check_valid(cloud):
                            clouds.append(cloud)

            logging.debug('Filtered clouds by visible area border')

        return clouds

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

            M = cv.moments(self.contour)
            self.center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

            self.contour_perimeter = cv.arcLength(self.contour, True)
            self.contour_area = cv.contourArea(self.contour)
            self.hull = cv.convexHull(self.contour)
            self.hull_perimeter = cv.arcLength(self.hull, True)
            self.hull_area = cv.contourArea(self.hull)

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

        def std(self) -> tuple:
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

        # TODO docstring usage example
        def edges(self,
                  in_steps: int,
                  out_steps: int,
                  convex_hull: bool = False,
                  appr_dist: int = DEFAULT_APPR_DIST,
                  step_len: float = DEFAULT_STEP_LEN) -> np.ndarray:
            """Gets samples of the surroundings of the contour of the
            cloud.

            First, perpendicular vectors to the contour are created. Those are
            estimated by computing the tangents for each pixel on the contour.
            Because pixels are discrete, they are actually secants from the
            pixel to the appr_dist-th one next to it. From those vectors the
            perpendicular vectors with length step_len are gained. For each
            perpendicular vector the span is computed i.e. an array of points
            which are a multiple of the original vector. Note that in_steps is
            the number of steps the span reaches into cloud from the boundary
            and out_steps the number of steps out of the cloud. These steps
            have the length step_len. Then the spans which lie fully inside the
            visible area are returned.


            Args:
                in_steps:
                The number of steps the span reaches into cloud from the
                boundary excluding the boundary itself.

                out_steps:
                The number of steps the span reaches out of the cloud from
                the boundary excluding the boundary itself.

                convex_hull:
                Whether the spans should be computed with the actual contour or
                with the convex hull.

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
                (number of valid spans, in_steps + out_steps + 1, 3),
                which represents the surroundings of the edge of the cloud.
            """

            contour = self.hull if convex_hull else self.contour

            appr_vec = np.roll(contour, appr_dist, axis=0)
            appr_vec -= contour
            appr_vec_norm = np.linalg.norm(appr_vec, axis=1)
            appr_vec = appr_vec * step_len / np.tile(appr_vec_norm, (2, 1)).T

            perp_vec = np.roll(appr_vec, 1, axis=1)
            perp_vec[:, 0] *= -1

            span_range = np.arange(-in_steps, out_steps + 1)[:, np.newaxis]
            spans = [np.matmul(span_range, vec[np.newaxis]) + contour[n]
                     for n, vec in enumerate(perp_vec)]
            spans = np.floor(np.array(spans)).astype('int')

            valid_spans = [span for span in spans if
                           np.all(span[:, 0] > 0)
                           and np.all(span[:, 0] < self.height)
                           and np.all(span[:, 1] > 0)
                           and np.all(span[:, 1] < self.width)]
            valid_spans = np.array(valid_spans)

            if not valid_spans.size:
                logging.info('The cloud has no valid spans')

            edges = np.array([[self.orig[point[0], point[1]]
                               for point in span]
                              for span in valid_spans])

            return edges

        def diff_edges(self,
                       in_dist: float,
                       out_dist: float,
                       convex_hull: bool = False,
                       appr_dist: int = DEFAULT_APPR_DIST,
                       step_len: float = DEFAULT_STEP_LEN) -> np.ndarray:
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
                               convex_hull=convex_hull,
                               appr_dist=appr_dist,
                               step_len=step_len)
            if not edges.size:
                return np.array([])

            return np.abs(np.diff(np.mean(edges, axis=0), axis=0)) / step_len

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
                 num_clouds: int,
                 border_threshold: float = 0.1):
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
        assert 0 == 1  # for debugging
        self.contours = self._get_contours()
        self.clouds = self._get_clouds(self.contours)

    # TODO update documentation accordingly to the cloud border changes

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
        mask, _ = cloud_filter.evaluate_image(self.orig)
        if not np.any(mask):
            raise ValueError('Orig has no clouds.')

        return cv.resize(mask, (self.width, self.height))

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
        contours = [np.squeeze(contour) for contour in contours]

        return contours

    def _get_clouds(self, contours) -> list:
        """Creates a list of clouds sorted by area from large to small."""

        clouds = []
        for contour in contours:
            mask = np.zeros((self.height, self.width), np.uint8)
            cv.drawContours(mask, [contour], 0, (255, 255, 255), -1)
            img = cv.bitwise_and(self.orig, self.orig, mask=mask)
            clouds.append(self.Cloud(self.orig, img, mask, contour))
        return sorted(clouds, key=lambda x: x.contour_area, reverse=True)

    # TODO combine/write those methods according to the comments below

    def _get_valid_clouds(self,
                          clouds: list,
                          min_size: int,
                          border_width: int,
                          contrast_threshold: float):

        # TODO instead of filtering by the border proportion on the circle
        #  (because it doesn't fit) get the border contrast by the edges method
        #  and discard the ones with a too high contrast
        #  sort first the clouds by size then apply edges until the
        #  threshold is reached

        def get_contrast():
            for cloud in clouds:
                out = cloud.mean_diff_edges()

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
                  appr_dist: int = 3,
                  step_len: float = 2) -> np.ndarray:
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

            perp_vec = np.roll(appr_vec, 1, axis=1)
            perp_vec[:, 0] *= -1

            span_range = np.arange(-in_steps, out_steps + 1)[:, np.newaxis]
            spans = [np.matmul(span_range, vec[np.newaxis]) + self.contour[n]
                     for n, vec in enumerate(perp_vec)]
            spans = np.floor(np.array(spans)).astype('int')

            center = np.array([self.height / 2, self.width / 2])
            norms = [np.linalg.norm(span - center, axis=1)
                     for span in spans]

            valid_spans = [span for span in spans if
                           np.all(span[:, 0] > 0)
                           and np.all(span[:, 0] < self.height)
                           and np.all(span[:, 1] > 0)
                           and np.all(span[:, 1] < self.width)]

            valid_spans = np.array(valid_spans)

            edges = np.array([[self.orig[point[0], point[1]]
                               for point in span]
                              for span in valid_spans])

            return edges

        def mean_diff_edges(self,
                            in_steps: int,
                            out_steps: int,
                            appr_dist: int = 3,
                            step_len: float = 2) -> np.ndarray:
            """A measurement for the roughness of the edge of the cloud.

            The function computes an average roughness of the edge of the cloud
            by averaging the absolute difference of the pixels near the
            boundary in each channel. For a more detailed description of the
            functionality and the arguments see Cloud.edges.

            Returns:
                A numpy array with the average change from one pixel to another
                in the direction of the boundary. It has shape 3.
            """
            edges = self.edges(in_steps, out_steps, appr_dist,
                               step_len)

            return np.mean(
                np.abs(np.diff(np.mean(edges, axis=0), axis=0))) * step_len

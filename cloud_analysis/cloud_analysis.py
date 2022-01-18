import cv2 as cv
import numpy as np
import skimage
from cloud_detection.cloud_filter import CloudFilter


# for more detailed explanation of the methods
# see http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth06.pdf
# and h<ttp://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf


class Analysis:

    def __init__(self, orig, num_clouds, distance, num_glcm, c_dist):
        """

        :param orig: the image you want to analyse
        :param num_clouds: the number of largest clouds you want to analyse
        :param distance: the length of the offset vector for the GLCMs
        :param num_glcm: number of GLCM averaged with rotated offset vectors
        """

        self.orig = orig
        self.height, self.width, self.channels = self.orig.shape

        # in this part the intelligent mask for the clouds is created and the clouds are separated
        cloud_filter = CloudFilter()
        mask, _ = cloud_filter.evaluate_image(orig)
        mask_re = cv.resize(mask, (self.width, self.height))

        all_contours, _ = cv.findContours(cv.medianBlur(mask_re, 3), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        areas = [cv.contourArea(cnt) for cnt in all_contours]
        max_areas = np.sort(areas)[-num_clouds:]
        self.contours = [all_contours[np.where(areas == max_area)[0][0]] for max_area in max_areas]

        # this is a list from the num_clouds largest clouds which are objects of the type Cloud
        self.clouds = []
        for contour in self.contours:
            mask = np.zeros((self.height, self.width), np.uint8)
            cv.drawContours(mask, [contour], 0, (255, 255, 255), -1)
            img = cv.bitwise_and(self.orig, self.orig, mask=mask)
            self.clouds.append(self.Cloud(img, mask, contour, distance, num_glcm, c_dist))

    def __str__(self):
        # TODO more information?
        return f'dimensions: {self.orig.shape}\nnumber of clouds: {len(self.clouds)}'

    class Cloud:
        def __init__(self, img, mask, contour, distance, num_glcm, c_dist):
            """

            :param img: The image of the cloud. while the background is black only the cloud itself has color.
            :param contour: list of the points that describe the border of the cloud
            :param distance: the length of the offset vector for the GLCMs
            :param num_glcm: number of GLCMs averaged with rotated offset vectors
            """

            self.img = img
            self.mask = mask
            self.contour = contour

            self.shape = self.Shape(self.contour)
            self.texture = self.Texture(self.img, self.mask, distance, num_glcm, c_dist)

        def __str__(self):
            return f'dimensions: {self.img.shape}\nedge width: {self.edge_width()}'

        class Shape:

            def __init__(self, contour):
                self.contour = contour
                self.contour_perimeter = cv.arcLength(self.contour, True)
                self.contour_area = cv.contourArea(self.contour)
                self.hull = cv.convexHull(self.contour)
                self.hull_perimeter = cv.arcLength(self.hull, True)
                self.hull_area = cv.contourArea(self.hull)

            def __str__(self):
                out = [f'Shape Analysis:\n',
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
            def __init__(self, img, mask, distance, num_glcm, c_dist):
                self.img = img
                self.mask = mask
                self.grey = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

                # gray-level co-occurrence matrix
                angles = np.arange(0, 2 * np.pi, np.pi / num_glcm * 2)
                self.glcm = skimage.feature.graycomatrix(self.grey, [distance], angles, normed=True)[:, :, 0, :]
                self.glcm = np.mean(self.glcm, axis=2)
                self.glcm = self.glcm[1:, 1:]

                # gray-level distance statistics
                self.glds = [np.sum(self.glcm.diagonal(n) + np.sum(self.glcm.diagonal(-n))) for n in range(256)]
                self.glds = self.glds / np.sum(self.glds)

                self.contrast_img = np.zeros(self.grey.shape)

                for offset_x in range(-c_dist, c_dist + 1):
                    for offset_y in range(-c_dist, c_dist + 1):
                        self.contrast_img += np.abs(self.grey - np.roll(self.grey, (offset_x, offset_y), axis=(1, 0)))

                self.contrast_img = np.floor(np.divide(self.contrast_img, (2 * c_dist + 1) ** 2))

            def __str__(self):
                out = ['Texture Analysis:\n',
                       f'   mean: {self.mean()}',
                       f'   standard deviation: {self.std()}',
                       f'   contrast: {self.contrast()}',
                       f'   glds skewness: 0.25: {self.glds_skewness(0.25)}',
                       f'   glds skewness: 0.5: {self.glds_skewness(0.5)}',
                       f'   glds skewness: 0.75: {self.glds_skewness(0.75)}',
                       f'   transparency: {self.transparency()}', ]
                return '\n'.join(out)

            def dist(self):
                data = []
                for n in range(3):
                    channel = np.array(self.img[:, :, n].ravel())
                    data.append(channel[channel.nonzero()])
                return data

            def mean(self):
                return cv.mean(self.img, mask=self.mask)[:-1]

            def std(self):
                _, std = cv.meanStdDev(self.img, mask=self.mask)
                std = (std[0][0], std[1][0], std[2][0])
                return std

            def contrast(self):
                return np.sum(self.contrast_img) / np.count_nonzero(self.contrast_img)

            def glds_skewness(self, proportion):
                level = 0
                for i, val in enumerate(self.glds):
                    level += val
                    if level >= proportion:
                        return i

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

        def edge_width(self):
            return self.texture.contrast() / self.shape.convexity()

    # TODO interpretation

    # TODO contrast based on contour

    # TODO circle too smooth border

    # TODO small clouds (stratus) aren't recognised

    # TODO more comments

    # TODO make code more stable

import cv2 as cv
import numpy as np
from cloud_detection.cloud_filter import CloudFilter


# for more detailed explanation of the methods
# see http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth06.pdf
# and http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf


class Analysis:

    def __init__(self, orig, num_clouds):
        """

        :param orig: the image you want to analyse
        :param num_clouds: the number of largest clouds you want to analyse
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
            self.clouds.append(self.Cloud(self.orig, img, mask, np.squeeze(contour)))

    def __str__(self):
        return f'dimensions: {self.orig.shape}\nnumber of clouds: {len(self.clouds)}'

    class Cloud:
        def __init__(self, orig, img, mask, contour):
            """

            :param img: The image of the cloud. while the background is black only the cloud itself has color.
            :param contour: list of the points that describe the border of the cloud
            """

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
            def __init__(self, img, mask):
                self.img = img
                self.mask = mask
                self.grey = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

            def __str__(self):
                out = ['Texture Analysis:\n',
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
                return cv.mean(self.img, mask=self.mask)[:-1]

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

        def edges(self, sample_proportion, in_steps, out_steps, regression_distance=1, regression_length=1):
            print(f'contour shape: {self.contour.shape}')
            print(self.contour)
            num_sample_points = np.floor(self.contour.shape[0] * sample_proportion).astype('int')
            indices = np.random.randint(0, high=self.contour.shape[0] - 1 - regression_distance, size=num_sample_points)
            sample_points = self.contour[indices]
            print(f'sample_points shape: {sample_points.shape}')

            regression_vectors = self.contour[indices + regression_distance] - sample_points
            regression_vectors = regression_vectors * regression_length / np.linalg.norm(regression_vectors, axis=1)
            print(f'regression vectors shape: {regression_vectors.shape}')
            print(regression_vectors)

            perpendicular_vectors = np.roll(regression_vectors, 1, axis=1)
            perpendicular_vectors[:, 0] *= -1
            print(f'perpendicular vectors shape: {perpendicular_vectors.shape}')
            print(perpendicular_vectors)

            spans = [[np.floor((vector * t) + sample_points[n]).astype('int') for t in range(-in_steps, out_steps + 1)]
                     for n, vector in enumerate(perpendicular_vectors)]
            spans = np.array(spans).astype('int')
            print(f'spans shape: {spans.shape}')

            edges = np.array([self.orig[span[0], span[1]] for span in spans])
            print(edges.shape)
            return edges

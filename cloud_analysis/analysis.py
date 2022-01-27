import cv2 as cv
import numpy as np

from cloud_detection.cloud_filter import CloudFilter


# for a more detailed explanation of the methods
# see http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth06.pdf
# and http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf


class Analysis:

    # TODO convert to multiple functions
    def __init__(self, orig, num_clouds):
        self.orig = orig
        self.height, self.width, self.channels = self.orig.shape

        self.mask = self._get_mask()

        self.contours = self._get_contours(num_clouds)

        self.clouds = self._get_clouds()

    def __str__(self):
        return f'dimensions: {self.orig.shape}\nnumber of clouds: {len(self.clouds)}'

    def _get_mask(self):
        cloud_filter = CloudFilter()
        mask, _ = cloud_filter.evaluate_image(self.orig)
        return cv.resize(mask, (self.width, self.height))

    def _get_contours(self, num_clouds):
        # get the contours of the clouds
        all_contours, _ = cv.findContours(cv.medianBlur(self.mask, 3), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        all_contours = [np.squeeze(cnt) for cnt in all_contours]

        areas = [cv.contourArea(cnt) for cnt in all_contours]
        max_areas = np.sort(areas)[-num_clouds:]
        return [all_contours[np.where(areas == max_area)[0][0]] for max_area in max_areas]

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

        def edges(self, num_samples, in_steps, out_steps, regr_distance=3, regr_length=1):
            # dtime = time()
            # assert len(self.orig.shape) == 3

            regr_vectors = np.roll(self.contour, regr_distance, axis=0) - self.contour
            regr_vectors = regr_vectors * regr_length / np.tile(np.linalg.norm(regr_vectors, axis=1), (2, 1)).T
            # assert len(regr_vectors.shape) == 2
            # assert regr_vectors.shape[1] == 2

            perp_vectors = np.roll(regr_vectors, 1, axis=1)
            perp_vectors[:, 0] *= -1
            # assert len(perp_vectors.shape) == 2
            # assert perp_vectors.shape[1] == 2

            spans = [[np.floor((vector * t) + self.contour[n]).astype('int')
                      for t in range(-in_steps, out_steps + 1)]
                     for n, vector in enumerate(perp_vectors)]
            spans = np.array(spans)
            # assert len(spans.shape) == 3
            # assert spans.shape[1] == in_steps + out_steps + 1
            # assert spans.shape[2] == 2

            valid_spans = [span for span in spans if
                           np.all(np.logical_and(np.logical_and(self.height > span[:, 0], span[:, 0] >= 0),
                                                 np.logical_and(self.width > span[:, 1], span[:, 1] >= 0)))]

            valid_spans = np.array(list(valid_spans))
            indices = np.sort(np.floor(np.linspace(0, valid_spans.shape[0] - 1, num=num_samples)).astype('int'))
            # assert len(indices.shape) == 1
            # assert indices.shape[0] == num_samples

            sample_spans = valid_spans[indices]
            # assert len(sample_spans.shape) == 3
            # assert sample_spans.shape[0] == num_samples
            # assert sample_spans.shape[1] == in_steps + out_steps + 1
            # assert sample_spans.shape[2] == 2

            edges = np.array([[self.orig[point[0], point[1]] for point in span] for span in sample_spans])

            out = np.mean(np.diff(np.mean(edges, axis=0), axis=0), axis=0)
            # assert out.dtype == float
            # print(time() - dtime)
            return out

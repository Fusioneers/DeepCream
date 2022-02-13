import os.path

import cv2
import numpy as np
from matplotlib import pyplot as plt
# from pycoral.utils import edgetpu


# TODO write test
from tensorflow.python.keras.models import load_model

from DeepCream.constants import ABS_PATH


class CloudDetection:
    def __init__(self,
                 blur=3, h_min=0, h_max=179, s_min=0, s_max=50, v_min=145,
                 v_max=255, contrast=1, brightness=0,
                 weight_ai=0.5, binary_cloud_threshold=100, tpu_support=False):

        """

        Args:

        Optional Args:
            binary_cloud_threshold:
            The threshold (between 0 and 255) which determines if the pixel is
            part of a cloud.

            blur:
            The blur cv2.blur uses in cv_generate_image_mask to adjust the
            input image.

            contrast:
            The contrast cv2.convertScaleAbs uses in cv_generate_image_mask to
            adjust the input image.

            brightness:
            The brightness cv2.convertScaleAbs uses in cv_generate_image_mask
            to adjust the input image.

            h_min:
            The minimum hue for a pixel to be considered a cloud.

            h_max:
            The maximum hue for a pixel to be considered a cloud.

            s_min:
            The minimum saturation for a pixel to be considered a cloud.

            s_max:
            The maximum saturation for a pixel to be considered a cloud.

            v_min:
            The minimum value for a pixel to be considered a cloud.

            v_max:
            The maximum value for a pixel to be considered a cloud.

            weight_ai:
            The importance of the AI prediction (from 0 to 1), the higher the
            value the more importance.

            tpu_support:
            Whether the systems support a tpu (False by default).
        """

        # Set thresholds for cloud detection
        self.binaryCloudThreshold = binary_cloud_threshold

        # Set the contrast, brightness and blur which should be applied to the
        # image
        self.contrast = contrast
        self.brightness = brightness
        self.blur = blur

        # Set the min/max Hue, Saturation and Value for the image filtering
        self.hMin = h_min
        self.hMax = h_max
        self.sMin = s_min
        self.sMax = s_max
        self.vMin = v_min
        self.vMax = v_max

        # The weight of the AI prediction
        self.weightAi = weight_ai

        # Load the machine learning model
        self.HEIGHT = 192
        self.WIDTH = 256
        self.CHANNELS = 3

        if not tpu_support:
            self.interpreter = None
            self.model = load_model(os.path.join(ABS_PATH,
                                                 'DeepCream/cloud_detection/models/keras'))
        # else:
        #     self.model = None
        #     self.interpreter = edgetpu.make_interpreter(os.path.join(ABS_PATH, 'DeepCream/cloud_detection/models'
        #                                                                        '/tflite/model.tflite'))
        #     self.interpreter.allocate_tensors()
        #     self.input_details = self.interpreter.get_input_details()
        #     self.output_details = self.interpreter.get_output_details()

    def __load_image(self, image: np.ndarray):

        """

        Returns:
            normal:
            The image resized to (self.WIDTH, self.HEIGHT).

            scaled:
            The image resized to (self.WIDTH, self.HEIGHT) and scaled to have
            pixel values between 0 and 1.

        """

        normal = cv2.resize(image, (self.WIDTH, self.HEIGHT),
                            interpolation=cv2.INTER_AREA)

        scaled = normal.astype('float32')
        scaled /= 255.0

        return normal, scaled

    def __ai_generate_image_mask(self, image):

        """

        Args:
            image:
            The cloud image in the dimensions
            (self.WIDTH, self.HEIGHT, self.CHANNELS).

        Returns:
            The cloud mask (calculated by the AI) in the dimensions
            (self.WIDTH, self.HEIGHT, 1), with 0 representing a 0% probability
            for a cloud and 255 representing a 100% chance.

        """

        if self.model is not None:
            pred = (self.model.predict(np.array([image])).reshape(
                self.HEIGHT, self.WIDTH, 1))
            mask = cv2.normalize(
                pred, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            return mask
        elif self.interpreter is not None:
            self.interpreter.set_tensor(
                self.input_details[0]['index'], [image])
            self.interpreter.invoke()
            pred = self.interpreter.get_tensor(
                self.output_details[0]['index'])
            mask = cv2.normalize(
                pred, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            return mask
        else:
            raise ValueError('No AI was configured')

    def __cv_generate_image_mask(self, image):

        """

        Args:
            image:
            The cloud image in rgb format.

        Returns:
            The cloud mask (calculated by OpenCV) in grayscale format.

        """

        # Blur the image
        if self.blur > 1:
            blurred = cv2.blur(image, (self.blur, self.blur))
        else:
            blurred = image

        # Adjust contrast and brightness
        adjusted = cv2.convertScaleAbs(blurred, alpha=self.contrast,
                                       beta=self.brightness)

        # Set lower and upper boundaries to filter out
        lower = np.array([self.hMin, self.sMin, self.vMin])
        upper = np.array([self.hMax, self.sMax, self.vMax])

        # Create the image mask to filter out everything but the clouds
        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
        rough_mask = cv2.inRange(hsv, lower, upper)
        hsv = cv2.bitwise_and(hsv, hsv, mask=rough_mask)

        # Create an empty mask with the right dimensions
        fine_mask = np.zeros(hsv.shape, hsv.dtype)

        # Calculate for each pixel how likely it is to be a cloud and five it
        # back as a mask
        for y in range(hsv.shape[0]):
            for x in range(hsv.shape[1]):
                # TODO RuntimeWarning: overflow encountered in ubyte_scalars ??
                hue = (hsv[y, x, 0] > 0) * np.clip(
                    255 - (hsv[:, :, 0].max() - hsv[y, x, 2]), 0, 255)

                saturation = (hsv[y, x, 1] > 0) * np.clip(255 - (hsv[y, x, 1]),
                                                          0, 255)

                value = np.clip(255 + hsv[y, x, 2] - self.vMax, 0, 255)

                fine_mask[y, x] = [hue, saturation, value]

        return cv2.cvtColor(fine_mask, cv2.COLOR_BGR2GRAY)

    def evaluate_image(self, image) -> np.ndarray:

        """

        Returns:
            A black and white mask with black representing no clouds and white
            representing clouds.

        """

        # Load the image
        normal, scaled = self.__load_image(image)

        # Check if the image actually loaded
        assert normal is not None and scaled is not None

        # Compute the two masks
        ai_mask = self.__ai_generate_image_mask(scaled).reshape(self.HEIGHT,
                                                                self.WIDTH)
        cv_mask = self.__cv_generate_image_mask(normal)

        plt.figure(figsize=(12, 8))
        plt.subplot(121)
        plt.title('cv_mask')
        plt.imshow(cv_mask)
        plt.subplot(122)
        plt.title('ai_mask')
        plt.imshow(ai_mask)
        plt.show()

        # Combine the two masks
        mask = cv2.addWeighted(ai_mask, self.weightAi, cv_mask,
                               (1 - self.weightAi), 0.0)

        # Normalize the resulting maks then make it binary
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        ret, mask = cv2.threshold(mask, self.binaryCloudThreshold, 255,
                                  cv2.THRESH_BINARY)

        # Apply the mask to filter out everything but the clouds
        # multi_color_output = cv2.bitwise_and(normal, normal, mask=mask)

        return mask

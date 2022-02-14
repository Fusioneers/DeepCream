import os
import cv2 as cv
import matplotlib.pyplot as plt
import shutil
from PIL import Image
# from DeepCream.cloud_detection.cloud_filter import CloudFilter
import numpy as np
# import faiss
import time as t
from scipy import mean
import operator

# from DeepCream.cloud_analysis.analysis import Analysis
from typing import Tuple, Dict, Any

from DeepCream.constants import ABS_PATH


# TODO __init__ has changed, see in the docstring of analysis


class Art:
    data_path = os.path.join(ABS_PATH, "data")
    art_data_path = os.path.join(data_path, "art-data")
    source = os.path.join(art_data_path, "MPEG-7")
    inp_dir = os.path.join(data_path, "output")

    def __init__(self):
        pass
        # self.data_path = os.path.join(ABS_PATH, "data")
        # self.art_data_path = os.path.join(self.data_path, "art-data")
        # self.source = os.path.join(self.art_data_path, "MPEG-7")
        # self.inp_dir = os.path.join(self.data_path, "output")

    def sort_files(self, filetype, divider):
        for img_name in os.listdir(self.source):
            if img_name.rsplit(".")[-1] == filetype:
                img_path = os.path.join(self.source, img_name)
                obj_name = img_name.rsplit(divider)[0]

                obj_path = os.path.join(self.dest, obj_name)

                if not os.path.isdir(obj_path):
                    os.mkdir(obj_path)

                dest_img_path = os.path.join(obj_path, img_name)
                shutil.copyfile(img_path, dest_img_path)

    # sort_files("gif", "-")

    def main(self):
        all_images = {}
        for inp_img in os.listdir(self.inp_dir):
            one_image = self.image_classification(inp_img)
            all_images.update({inp_img: one_image})
        return all_images

    def image_classification(self, inp_img):
        results = []

        img_path = os.path.join(self.inp_dir, inp_img)

        binary_all_clouds = cv.imread(img_path)

        clouds_cnts, _ = cv.findContours(binary_all_clouds[:, :, 0],
                                         cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

        for cnt in clouds_cnts:
            if cv.contourArea(cnt) > 100:  # TODO adjust
                one_cloud = self.shape_classification(cnt, binary_all_clouds.shape)
                results.append(one_cloud)

        return results

    def shape_classification(self, cnt, img_shape):
        mask = np.zeros(img_shape, "uint8")
        cv.drawContours(mask, [cnt], -1, (255, 255, 255), -1)
        cv.imshow("shape", mask)

        scoring = self.get_scores(cnt)
        print(scoring)

        result = self.interpret_scoring(scoring)
        best = min(result.items(), key=operator.itemgetter(1))[0]
        print(result)
        print(best)

        cv.waitKey(0)
        return result
        # try:

    def get_scores(self, inp_cnt) -> Dict[str, Dict[str, float]]:
        # mask = np.zeros((inp_img.shape[0], inp_img.shape[1]) , np.uint8)+
        # cv.drawContours(mask, [cnt], 0, (255, 255, 255), -1)
        # cv.imshow("test", mask)
        # cv.waitKey()
        scores = {}
        string_out = ""
        for img_from_mpeg in os.listdir(self.source):
            img_from_mpeg_path = os.path.join(self.source, img_from_mpeg)
            cap = cv.VideoCapture(img_from_mpeg_path)
            ret, fm = cap.read()
            cap.release()
            fm_contours, _ = cv.findContours(fm[:, :, 0],
                                             cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

            predefined_cnt = fm_contours[0]
            # print(cnt)
            # cv.resize(fm_contours, (mask.shape[0], mask.shape[1]))

            score = cv.matchShapes(inp_cnt, predefined_cnt, 2, 0.0)

            if score < 0.05:
                string_out += f"{str(score)} {img_from_mpeg}\n"

            shape_group_name = img_from_mpeg.rsplit("-")[0]

            try:
                scores[shape_group_name].update({img_from_mpeg: score})
            except KeyError:
                scores.update({shape_group_name: {img_from_mpeg: score}})

            # print(f"Predefined: {img_from_mpeg}\nScore:      {score}")

        # print(string_out)  # TODO remove "string_out"

        return scores

    def interpret_scoring(self, scoring) -> Dict[Any, float]:
        results = {}

        for predefined_shape_group in scoring:
            average = sum(scoring[predefined_shape_group].values()) / len(scoring[predefined_shape_group])
            results.update({predefined_shape_group: average})

        return results


art = Art()
out = art.main()

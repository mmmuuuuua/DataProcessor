import cv2
import numpy as np
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
import torch.nn as nn
import os


def clip_label(input_path, output_path):
    for img_path in os.listdir(input_path):
        img = cv2.imread(os.path.join(input_path, img_path), 0)
        h, w = img.shape
        print(h, w)
        resize_h = h // 2
        resize_w = w // 2

        img1 = img[0:resize_h, 0:resize_w]
        img2 = img[resize_h:h, 0:resize_w]
        img3 = img[0:resize_h, resize_w:w]
        img4 = img[resize_h:h, resize_w:w]

        cv2.imwrite(os.path.join(output_path, os.path.splitext(img_path)[0] + "_1" + os.path.splitext(img_path)[1]), img1)
        cv2.imwrite(os.path.join(output_path, os.path.splitext(img_path)[0] + "_2" + os.path.splitext(img_path)[1]), img2)
        cv2.imwrite(os.path.join(output_path, os.path.splitext(img_path)[0] + "_3" + os.path.splitext(img_path)[1]), img3)
        cv2.imwrite(os.path.join(output_path, os.path.splitext(img_path)[0] + "_4" + os.path.splitext(img_path)[1]), img4)


def clip_image(input_path, output_path):
    for img_dir_path in os.listdir(input_path):
        for i in range(1, 5):
            out_img_dir_path = os.path.join(output_path, img_dir_path + "_" + str(i))
            if not os.path.exists(out_img_dir_path):
                os.mkdir(out_img_dir_path)

        for img_path in os.listdir(os.path.join(input_path, img_dir_path)):
            img = cv2.imread(os.path.join(input_path, img_dir_path, img_path))
            h, w, c = img.shape
            resize_h = h // 2
            resize_w = w // 2

            img1 = img[0:resize_h, 0:resize_w]
            img2 = img[resize_h:h, 0:resize_w]
            img3 = img[0:resize_h, resize_w:w]
            img4 = img[resize_h:h, resize_w:w]

            cv2.imwrite(os.path.join(output_path, img_dir_path + "_1", img_path), img1)
            cv2.imwrite(os.path.join(output_path, img_dir_path + "_2", img_path), img2)
            cv2.imwrite(os.path.join(output_path, img_dir_path + "_3", img_path), img3)
            cv2.imwrite(os.path.join(output_path, img_dir_path + "_4", img_path), img4)


def main():
    clip_label("D:\\zl\\GraduationThesis\\data\\new_data\\label",
               "D:\\zl\\GraduationThesis\\data\\new_data\\crop_label")
    clip_image("D:\\zl\\GraduationThesis\\data\\new_data\\image",
               "D:\\zl\\GraduationThesis\\data\\new_data\\crop_image")


if __name__ == '__main__':
    main()



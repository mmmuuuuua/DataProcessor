import random
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def process(input_path, output_path):
    for f in os.listdir(input_path):

        img = cv2.imread(os.path.join(input_path, f), 0)

        ret, binary = cv2.threshold(img, 75, 255, cv2.THRESH_BINARY)

        ret2, binary2 = cv2.threshold(img, 179, 255, cv2.THRESH_BINARY)

        # cv2.imshow('binary', binary)
        # cv2.waitKey(0)
        #
        # cv2.imshow('binary2', binary2)
        # cv2.waitKey(0)

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(binary2, contours, -1, 255, 1)

        # cv2.imshow('binary2', binary2)
        # cv2.waitKey(0)

        cv2.imwrite(os.path.join(output_path, f), binary2)


if __name__ == '__main__':
    input = 'C:\\zhulei\\gsn\\Pytorch-UNet-Sequence\\data\\train\\labels'
    output = "C:\\zhulei\\gsn\\Pytorch-UNet-Sequence\\data\\train\\edges"
    process(input, output)
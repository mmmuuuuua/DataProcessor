import numpy as np
from PIL import Image
import os


def process(input_path):
    # for category in os.listdir(input_path):
    #     category_path = os.path.join(input_path, category)
    #     all_paths = os.listdir(category_path)
    #     all_paths.sort()
    #     for i, path in enumerate(all_paths, 1):
    #         srcFile = os.path.join(category_path, path)
    #         dstFile = os.path.join(category_path, category + str(i))
    #         os.rename(srcFile, dstFile)
    # for i in range(66, 66 + 83):
    #     os.mkdir(os.path.join(input_path, "PLA"+str(i)))
    a = 0
    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    c5 = 0
    c6 = 0
    c7 = 0
    c8 = 0
    c9 = 0
    for img_dir in os.listdir(input_path):
        img_dir_path = os.path.join(input_path, img_dir)
        for img in os.listdir(img_dir_path):
            if img.find('A') != -1:
                os.rename(os.path.join(img_dir_path, img), os.path.join(img_dir_path, "A1.jpg"))
                a = a + 1
            elif img.find('C1') != -1:
                os.rename(os.path.join(img_dir_path, img), os.path.join(img_dir_path, "C1.jpg"))
                c1 = c1 + 1
            elif img.find('C2') != -1:
                os.rename(os.path.join(img_dir_path, img), os.path.join(img_dir_path, "C2.jpg"))
                c2 = c2 + 1
            elif img.find('C3') != -1:
                os.rename(os.path.join(img_dir_path, img), os.path.join(img_dir_path, "C3.jpg"))
                c3 = c3 + 1
            elif img.find('C4') != -1:
                os.rename(os.path.join(img_dir_path, img), os.path.join(img_dir_path, "C4.jpg"))
                c4 = c4 + 1
            elif img.find('C5') != -1:
                os.rename(os.path.join(img_dir_path, img), os.path.join(img_dir_path, "C5.jpg"))
                c5 = c5 + 1
            elif img.find('C6') != -1:
                os.rename(os.path.join(img_dir_path, img), os.path.join(img_dir_path, "C6.jpg"))
                c6 = c6 + 1
            elif img.find('C7') != -1:
                os.rename(os.path.join(img_dir_path, img), os.path.join(img_dir_path, "C7.jpg"))
                c7 = c7 + 1
            elif img.find('C8') != -1:
                os.rename(os.path.join(img_dir_path, img), os.path.join(img_dir_path, "C8.jpg"))
                c8 = c8 + 1
            elif img.find('C9') != -1:
                os.rename(os.path.join(img_dir_path, img), os.path.join(img_dir_path, "C9.jpg"))
                c9 = c9 + 1

    print(a)
    print(c1)
    print(c2)
    print(c3)
    print(c4)
    print(c5)
    print(c6)
    print(c7)
    print(c8)
    print(c9)


if __name__ == '__main__':
    input_path = "D:\\zl\\GraduationThesis\\data\\classification_data\\classification_data\\QUA"
    process(input_path)
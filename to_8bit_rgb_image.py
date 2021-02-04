import numpy as np
from PIL import Image
import os


def process(input_path, output_path):
    for f in os.listdir(input_path):
        img = Image.open(os.path.join(input_path, f))
        img.show()
        convert_img = img.convert("P")
        print(convert_img.mode)
        convert_img.save(os.path.join(output_path, f))
        convert_img.show()


if __name__ == '__main__':
    input_path = 'D:\\zl\\GraduationThesis\\data\\new_data\\label3'
    output_path = 'D:\\zl\\GraduationThesis\\data\\new_data\\label3RGB'
    process(input_path, output_path)
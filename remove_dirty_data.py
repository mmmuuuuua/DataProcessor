import numpy as np
from PIL import Image
import os


def process(segmentation_dir, sequence_dir):
    for f in os.listdir(segmentation_dir):
        mask = np.array(Image.open(os.path.join(segmentation_dir, f)))
        num = np.max(mask)
        if num == 0:
            os.remove(os.path.join(segmentation_dir, f))
            for sub_file in os.listdir(os.path.join(sequence_dir, os.path.splitext(f)[0])):
                os.remove(os.path.join(os.path.join(sequence_dir, os.path.splitext(f)[0]), sub_file))
            os.rmdir(os.path.join(sequence_dir, os.path.splitext(f)[0]))


if __name__ == '__main__':
    segmentation_object = 'C:\\zhulei\\maskRcnn\\data\\train\\SegmentationObject'
    sequence_images = 'C:\\zhulei\\maskRcnn\\data\\train\\SequenceImages'
    process(segmentation_object, sequence_images)
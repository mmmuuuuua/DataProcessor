import glob
import os
from pycocotools.coco import COCO
import cv2
import numpy as np
from PIL import Image

annFile = "C:\\zhulei\\data\\instance_segmentation\\rock2\\shapes\\train\\instances_leaf_train2017.json"
root = "C:\\zhulei\\data\\instance_segmentation\\rock2\\shapes\\train"


if __name__ == '__main__':
        coco = COCO(annFile)
        ids = list(sorted(coco.imgs.keys()))
        print(ids)
        img_id = ids[1]

        print(img_id)

        ann_ids = coco.getAnnIds(imgIds=img_id)

        print(ann_ids)

        target = coco.loadAnns(ann_ids)

        # print(target)

        path = coco.loadImgs(img_id)[0]['file_name']

        print(path)

        img = Image.open(os.path.join(root, path)).convert('RGB')
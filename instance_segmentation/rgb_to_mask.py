import glob
import os

import cv2
import numpy as np

root = "D:\\zl\\GraduationThesis\\data\\new_data\\coco_format"


def rgb_to_mask(label_name):
    lbl_id = os.path.split(label_name)[-1].split('.')[0]
    lbl = cv2.imread(label_name, 0)
    h, w = lbl.shape[:2]
    leaf_dict = {}
    idx = 0
    white_mask = np.ones((h, w), dtype=np.uint8) * 255
    for i in range(h):
        for j in range(w):
            if lbl[i][j] in leaf_dict or lbl[i][j] == 0:
                continue
            leaf_dict[lbl[i][j]] = idx
            mask = (lbl == lbl[i][j])
            # leaf = lbl * mask[..., None]      # colorful leaf with black background
            # np.repeat(mask[...,None],3,axis=2)    # 3D mask
            leaf = np.where(mask, white_mask, 0)
            dir_name = os.path.join(root, 'shapes\\train\\annotations\\', lbl_id)
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            # mask_name = os.path.join(root, 'shapes\\train\\annotations\\') + lbl_id + '_rock_' + str(idx) + '.png'
            mask_name = lbl_id + '_rock_' + str(idx) + '.png'
            cv2.imwrite(os.path.join(dir_name, mask_name), leaf)
            idx += 1


if __name__ == '__main__':
    label_dir = os.path.join(root, 'shapes\\crop_label3')
    label_list = glob.glob(os.path.join(label_dir, '*.BMP'))
    for label_name in label_list:
        rgb_to_mask(label_name)
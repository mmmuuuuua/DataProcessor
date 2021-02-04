import numpy as np
from PIL import Image
import os


def bfs(single_img_mask, sequence_img_mask, i, j, vis, val):
    q = [(i, j)]
    vis[i][j] = False
    cnt = 0 if sequence_img_mask[i][j] == 0 else 1
    total = 1
    for i, j in q:
        for x, y in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= x < single_img_mask.shape[0] and 0 <= y < single_img_mask.shape[1] and single_img_mask[x][y] != 0 and vis[x][y] == True:
                if sequence_img_mask[x][y] != 0:
                    cnt = cnt + 1
                total = total + 1
                vis[x][y] = False
                q.append((x, y))

    return cnt / total


def clean(mask, i, j):
    q, mask[i][j] = [(i, j)], 0
    for i, j in q:
        for x, y in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] and mask[x][y] > 0:
                mask[x][y] = 0
                q.append((x, y))
    return 1


def fusion(single_img_mask, sequence_img_mask):
    vis = np.ones((single_img_mask.shape[0], single_img_mask.shape[1]), dtype=bool)
    for i in range(single_img_mask.shape[0]):
        for j in range(single_img_mask.shape[1]):
            if single_img_mask[i][j] != 0 and vis[i][j] == True:
                t = bfs(single_img_mask, sequence_img_mask, i, j, vis, single_img_mask[i][j])
                if t > 0.2:
                    clean(single_img_mask, i, j)

    for i in range(single_img_mask.shape[0]):
        for j in range(single_img_mask.shape[1]):
            if single_img_mask[i][j] != 0:
                sequence_img_mask[i][j] = 122

    return sequence_img_mask


def process(single_img_dir, sequence_img_dir, output_dir):
    for f in os.listdir(single_img_dir):
        single_img_mask = np.array(Image.open(os.path.join(single_img_dir, f)))
        sequence_img_mask = np.array(Image.open(os.path.join(sequence_img_dir, f)))
        fusion_img = fusion(single_img_mask, sequence_img_mask)
        Image.fromarray(fusion_img).save(os.path.join(output_dir, f))


if __name__ == '__main__':
    single_img_dir = 'D:\\zl\\GraduationThesis\\data\\new_data\\label1'
    sequence_img_dir = 'D:\\zl\\GraduationThesis\\data\\new_data\\label2'
    output_dir = 'D:\\zl\\GraduationThesis\\data\\new_data\\label2'
    process(single_img_dir, sequence_img_dir, output_dir)
import numpy as np
from PIL import Image
import os


area_thres = 444


def count(mask, i, j, vis, val):
    pos = np.where(mask == val)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    cnt = 1
    q = [(i, j)]
    vis[i][j] = False
    for i, j in q:
        for x, y in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] and mask[x][y] == 0 and vis[x][y] == True:
                vis[x][y] = False
                q.append((x, y))
                cnt = cnt + 1
    return xmax - xmin, ymax - ymin, cnt


def clean(mask, i, j):
    q, mask[i][j] = [(i, j)], 0
    for i, j in q:
        for x, y in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] and mask[x][y] == 0:
                mask[x][y] = 0
                q.append((x, y))
    return 1


def reduction(mask):
    vis = np.ones((mask.shape[0], mask.shape[1]), dtype=bool)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 0 and vis[i][j] == True:
                w, h, cnt = count(mask, i, j, vis, mask[i][j])
                if cnt <= area_thres or w <= 0 or h <= 0:
                    clean(mask, i, j)


def process(input, output):
    print(len(os.listdir(input)))
    for f in os.listdir(input):
        print(os.path.join(input, f))
        mask = np.array(Image.open(os.path.join(input, f)))
        reduction(mask)
        print(os.path.join(output, f))
        Image.fromarray(mask).save(os.path.join(output, f))


if __name__ == '__main__':
    input = 'D:\\zl\\GraduationThesis\\material\\test\\convlstm_9'
    output = 'D:\\zl\\GraduationThesis\\material\\test\\convlstm_999'
    process(input, output)
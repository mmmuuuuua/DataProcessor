import numpy as np
from PIL import Image
import os


def bfs(mask, i, j, vis, val):
    q = [(i, j)]
    vis[i][j] = False
    for i, j in q:
        for x, y in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] and mask[x][y] != [0, 0, 0] and vis[x][y] == True:
                if mask[x][y] != mask[i][j]:
                    mask[i][j] = 0
                else:
                    vis[x][y] = False
                    q.append((x, y))


def reduction(mask):
    print(mask.shape)
    vis = np.ones((mask.shape[0], mask.shape[1]), dtype=bool)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] != [0, 0, 0] and vis[i][j] == True:
                bfs(mask, i, j, vis, mask[i][j])


def process(input, output):
    print(len(os.listdir(input)))
    for f in os.listdir(input):
        print(os.path.join(input, f))
        mask = np.array(Image.open(os.path.join(input, f)))
        reduction(mask)

        output_img = np.ones((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                output_img[i][j] = 0 if mask[i][j] == [0, 0, 0] else 255

        print(os.path.join(output, f))
        Image.fromarray(output_img).save(os.path.join(output, f))


if __name__ == '__main__':
    input = 'D:\\zl\\GraduationThesis\\data\\new_data\\label1'
    output = 'D:\\zl\\GraduationThesis\\data\\new_data\\label2'
    process(input, output)
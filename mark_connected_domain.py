import numpy as np
from PIL import Image
import os

area_thres = 0
label_color = 249


def bfs(mask, i, j, color, vis):
    vis[i][j] = False
    q, mask[i][j] = [(i, j)], color
    for i, j in q:
        for x, y in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] and mask[x][y] > 0 and vis[x][y] == True:
                vis[x][y] = False
                mask[x][y] = color
                q.append((x, y))
    return 1


# def dfs(mask, x, y, color):
#     if x < 0 or y < 0 or x >= mask.shape[0] or y >= mask.shape[1] or mask[x][y] != label_color:
#         return
#
#     print(x, y)
#     mask[x][y] = color
#     dfs(mask, x + 1, y, color)
#     dfs(mask, x, y + 1, color)
#     dfs(mask, x - 1, y, color)
#     dfs(mask, x, y - 1, color)


def mark_every_connected_domain(mask):
    color = 1
    vis = np.ones((mask.shape[0], mask.shape[1]), dtype=bool)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] > 0 and vis[i][j] == True:
                bfs(mask, i, j, color, vis)
                color = color + 1
    print("color is: {}".format(color))


def process(input, output):
    print(len(os.listdir(input)))
    for f in os.listdir(input):
        print(os.path.join(input, f))
        mask = np.array(Image.open(os.path.join(input, f)))
        mark_every_connected_domain(mask)
        print(os.path.join(output, f))
        Image.fromarray(mask).save(os.path.join(output, f))


if __name__ == '__main__':
    input = 'D:\\zl\\GraduationThesis\\data\\new_data\\label2'
    output = 'D:\\zl\\GraduationThesis\\data\\new_data\\label3'
    process(input, output)
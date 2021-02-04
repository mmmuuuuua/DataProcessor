#pytorch批处理裁剪图像
# coding: utf-8
from PIL import Image
import os
import os.path
import numpy as np
import cv2
# 指明被遍历的文件夹

# rootdir = "D:\\zl\\GraduationThesis\\data\\new_data\\label"


# 裁剪单偏光、融合、标签图
# for parent, dirnames, filenames in os.walk(rootdir):  # 遍历每一张图片
#     for filename in filenames:
#         (name, extension) = os.path.splitext(filename)
#         for i in range(1,19):  #每张原图为1824*1216  裁剪成18张512*512  有重叠位置
#             print('parent is :' + parent)
#             print('filename is :' + filename)
#             currentPath = os.path.join(parent, filename)
#             print('the fulll name of the file is :' + currentPath)
#             img = Image.open(currentPath)
#             print(img.format, img.size, img.mode)
#             # img.show()
#             if i == 1:
#                 box1 = (0, 0, 512, 512)  # 设置左、上、右、下的像素
#             elif i == 2:
#                 box1 = (256, 0, 768, 512)  # 设置左、上、右、下的像素
#             elif i == 3:
#                 box1 = (512, 0, 1024, 512)  # 设置左、上、右、下的像素
#             elif i == 4:
#                 box1 = (768, 0, 1280, 512)  # 设置左、上、右、下的像素
#             elif i == 5:
#                 box1 = (1024, 0, 1536, 512)  # 设置左、上、右、下的像素
#             elif i == 6:
#                 box1 = (1280, 0, 1792, 512)  # 设置左、上、右、下的像素
#             elif i == 7:
#                 box1 = (0, 256, 512, 768)  # 设置左、上、右、下的像素
#             elif i == 8:
#                 box1 = (256, 256, 768, 768)  # 设置左、上、右、下的像素
#             elif i == 9:
#                 box1 = (512, 256, 1024, 768)  # 设置左、上、右、下的像素
#             elif i == 10:
#                 box1 = (768, 256, 1280, 768)  # 设置左、上、右、下的像素
#             elif i == 11:
#                 box1 = (1024, 256, 1536, 768)  # 设置左、上、右、下的像素
#             elif i == 12:
#                 box1 = (1280, 256, 1792, 768)  # 设置左、上、右、下的像素
#             elif i == 13:
#                 box1 = (0, 512, 512, 1024)  # 设置左、上、右、下的像素
#             elif i == 14:
#                 box1 = (256, 512, 768, 1024)  # 设置左、上、右、下的像素
#             elif i == 15:
#                 box1 = (512, 512, 1024, 1024)  # 设置左、上、右、下的像素
#             elif i == 16:
#                 box1 = (768, 512, 1280, 1024)  # 设置左、上、右、下的像素
#             elif i == 17:
#                 box1 = (1024, 512, 1536, 1024)  # 设置左、上、右、下的像素
#             else:# i == 18:
#                 box1 = (1280, 512, 1792, 1024)  # 设置左、上、右、下的像素
#
#             image1 = img.crop(box1)  # 图像裁剪
#             image1.save(os.path.join("D:\\zl\\GraduationThesis\\data\\new_data\\semantic_label",
#                                      name + '-' + str(i) + extension))
#             print('the file ' + filename + ' be croped successed! ')

# 指明被遍历的文件夹

rootdir = "D:\\zl\\GraduationThesis\\data\\new_data\\image"
outputdir = "D:\\zl\\GraduationThesis\\data\\new_data\\semantic_image"

#裁剪序列图
for subfolder in os.listdir(rootdir):  # 遍历母文件夹图片
    subpath = rootdir + '\\' +subfolder
    print('subfolder name is:', subpath)
    for filename in os.listdir(subpath):# 遍历序列图子文件夹中的图片
        (name, extension) = os.path.splitext(filename) #分离文件名和文件扩展名
        for i in range(1, 19):  #每张原图为1824*1216  裁剪成18张512*512  有重叠位置
            if not os.path.exists(os.path.join(outputdir, subfolder + "-" + str(i))):
                os.mkdir(os.path.join(outputdir, subfolder + "-" + str(i)))
            print('filename is :' + filename)
            currentPath = os.path.join(subpath, filename)
            print('the fulll name of the file is :' + currentPath)
            img = Image.open(currentPath)
            print(img.format, img.size, img.mode)
            # img.show()
            if i == 1:
                box1 = (0, 0, 512, 512)  # 设置左、上、右、下的像素
            elif i == 2:
                box1 = (256, 0, 768, 512)  # 设置左、上、右、下的像素
            elif i == 3:
                box1 = (512, 0, 1024, 512)  # 设置左、上、右、下的像素
            elif i == 4:
                box1 = (768, 0, 1280, 512)  # 设置左、上、右、下的像素
            elif i == 5:
                box1 = (1024, 0, 1536, 512)  # 设置左、上、右、下的像素
            elif i == 6:
                box1 = (1280, 0, 1792, 512)  # 设置左、上、右、下的像素
            elif i == 7:
                box1 = (0, 256, 512, 768)  # 设置左、上、右、下的像素
            elif i == 8:
                box1 = (256, 256, 768, 768)  # 设置左、上、右、下的像素
            elif i == 9:
                box1 = (512, 256, 1024, 768)  # 设置左、上、右、下的像素
            elif i == 10:
                box1 = (768, 256, 1280, 768)  # 设置左、上、右、下的像素
            elif i == 11:
                box1 = (1024, 256, 1536, 768)  # 设置左、上、右、下的像素
            elif i == 12:
                box1 = (1280, 256, 1792, 768)  # 设置左、上、右、下的像素
            elif i == 13:
                box1 = (0, 512, 512, 1024)  # 设置左、上、右、下的像素
            elif i == 14:
                box1 = (256, 512, 768, 1024)  # 设置左、上、右、下的像素
            elif i == 15:
                box1 = (512, 512, 1024, 1024)  # 设置左、上、右、下的像素
            elif i == 16:
                box1 = (768, 512, 1280, 1024)  # 设置左、上、右、下的像素
            elif i == 17:
                box1 = (1024, 512, 1536, 1024)  # 设置左、上、右、下的像素
            else:# i == 18:
                box1 = (1280, 512, 1792, 1024)  # 设置左、上、右、下的像素

            image1 = img.crop(box1)  # 图像裁剪
            # image1.save(r"G:\偏光图像数据集\2020.01.12新建数据集及测试集\序列图+单偏光-cut" +
            #             '\\'+'20200112'+'-'+subfolder+ '-'+str(i)+'\\'+'20200112'+'-'
            #             +subfolder+ '-'+str(i)+'-'+name+extension)
            image1.save(os.path.join(outputdir, subfolder + "-" + str(i), filename))
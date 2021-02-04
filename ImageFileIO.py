#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
import os
import numpy as np

def mkdirs(new_path):
    '''
    make Multilevel directory
    :param new_path:
    :return: Bool
    '''
    # remove the first space and the last '\'
    new_path = new_path.strip()
    new_path = new_path.rstrip('\\')
    # if the path has already existed
    is_exists = os.path.exists(new_path)
    if not is_exists:
        # if False, create new content
        os.makedirs(new_path)
        print(r'Create path: {} succeed.'.format(new_path))
        return True
    else:
        # if True
        print(r'the path: {} has already existed'.format(new_path))
        return False


def ReadImagesFromLabelFile(file_path, separator=','):
    # Take Dir Path
    file_path = file_path.replace('\\','/')
    dir_path = file_path[0:file_path.rfind('/')+1]

    # Open Image Label File and Read images
    images = []
    images_filepath = []
    with open(file_path, 'r') as image_label_file:
        images_path = image_label_file.readlines()
    for i in range(0, len(images_path)):
        images_path[i] = images_path[i].split(separator, 1)[0]
        if images_path[i][0] == '.':
            image_path = dir_path + images_path[i][2:]
        elif images_path[i][1] != ':':
            image_path = dir_path + images_path[i][:]
        if (image_path.rsplit('.', 1)[1] == "jpg" or image_path.rsplit('.', 1)[1] == "JPG") and os.path.exists(image_path):
            img = cv2.imread(image_path)
            images.append(img)
            images_filepath.append(images_path[i])
    return [images, dir_path, images_filepath]

def WriteImagesFromRelativePathList(images, dir_path, path_list):
    for i in range(0 ,len(images)):
        dir = path_list[i].rsplit('/', 1)[0]
        if dir[0] == '.':
            dir = dir_path + dir[2:]
        elif dir[0] != ':':
            dir = dir_path + dir
        if not os.path.exists(dir):
            mkdirs(dir)
        image_path = dir + '/' + path_list[i].rsplit('/', 1)[1]
        cv2.imwrite(image_path, images[i])

def ReadImagesFromDirectory(rootdirv):
    '''
    Read images from a root directory(Multilevel directory is surpoted).
    :param rootdirv: the path of rootdirectory
    :return:  list of all images and a list of all images path
    '''
    images = []
    images_filepath = []
    for parent, dirnames, filenames in os.walk(rootdirv):
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext != '.jpg' and ext != '.JPG':
                continue
            sourceFile = os.path.join(parent, filename)
            if not os.path.exists(sourceFile):
                continue
            images_filepath.append(sourceFile)
            images.append(cv2.imread(sourceFile))
    return [images, images_filepath]

def WriteImagesFromPathList(images, images_filepath):
    '''
    Write images accroding to the iamges_filepath.
    :param images: a list of images
    :param images_filepath: a list of images' path
    :return: True(All images are wrote) or False(Not all image are wrote)
    '''
    for i in range(0 ,len(images)):
        dir = os.path.split(images_filepath[i])[0]
        if not os.path.exists(dir):
            mkdirs(dir) # using mkdirs(dir) instead of os.makedirs(dir)
        cv2.imwrite(images_filepath[i], images[i])
    return True

def WriteLabel(rootdirv, labelTxtDirv=None, bShuffle=True, total=None):
    '''
    Write label for all images in rootdirv
    :param rootdirv: the images' root directory path
    :param labelTxtDirv: the labelTxt dir path, equal rootdirv if set None
    :param bShuffle: True or False for Shuffle
    :param total: the total number of each category, if you want align size, set this parameter
    :return: True of False
    '''
    if not labelTxtDirv:
        labelTxtDir = rootdirv
    if not (os.path.exists(rootdirv) or os.path.exists(labelTxtDirv)):
        return False
    all_label = list()
    num_label = {}
    for parent, dirnames, filenames in os.walk(rootdirv):
        labelId = os.path.basename(parent)
        if not labelId.isnumeric():
            continue
        num_label[labelId] = 0
        mix_label = list()
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext != '.jpg' and ext != '.JPG':
                continue
            sourceFile = os.path.join(parent, filename)
            if not os.path.exists(sourceFile):
                continue
            label = sourceFile.replace(labelTxtDirv, '').replace('\\', '/').lstrip('/') + ' ' + labelId + '\n'
            mix_label.append(label)
        if bShuffle:
            np.random.shuffle(mix_label)
        # align size, if total is not None
        if total and total < len(mix_label) + num_label[labelId]:
            assert num_label[labelId] <= total
            mix_label = mix_label[:total - num_label[labelId]]
        num_label[labelId] += len(mix_label)
        all_label.extend(mix_label)
    print(num_label)
    if bShuffle:
        np.random.shuffle(all_label)
    with open(r'%s\%s_label.txt' % (labelTxtDirv, os.path.basename(rootdirv)), 'w') as txt:
        txt.writelines(all_label)
        txt.close()
    return True

def WriteLabelEx(rootdirv, labelTxtDirv=None, bShuffle=True, numEveryone=None):
    '''
    Write label for all images in rootdirv, number of every directory are support
    :param rootdirv: the images' root directory path
    :param labelTxtDirv: the labelTxt dir path, equal rootdirv if set None
    :param bShuffle: True or False for Shuffle
    :param numEveryone: A map indicates the number of each category, if you want align size, set this parameter
    :return: True of False
    '''
    if not labelTxtDirv:
        labelTxtDir = rootdirv
    if not (os.path.exists(rootdirv) or os.path.exists(labelTxtDirv)):
        return False
    all_label = list()
    num_label = {}
    for parent, dirnames, filenames in os.walk(rootdirv):
        labelId = os.path.basename(parent)
        if not labelId.isnumeric():
            continue
        num_label[labelId] = 0
        mix_label = list()
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext != '.jpg' and ext != '.JPG':
                continue
            sourceFile = os.path.join(parent, filename)
            if not os.path.exists(sourceFile):
                continue
            label = sourceFile.replace(labelTxtDirv, '').replace('\\', '/').lstrip('/') + ' ' + labelId + '\n'
            mix_label.append(label)
        if bShuffle:
            np.random.shuffle(mix_label)
        # align size, if numEveryone is not None
        if numEveryone[labelId] and numEveryone[labelId] < len(mix_label) + num_label[labelId]:
            assert num_label[labelId] <= numEveryone[labelId]
            mix_label = mix_label[:numEveryone[labelId] - num_label[labelId]]
        num_label[labelId] += len(mix_label)
        all_label.extend(mix_label)
    print(num_label)
    if bShuffle:
        np.random.shuffle(all_label)
    with open(r'%s\%s_label.txt' % (labelTxtDirv, os.path.basename(rootdirv)), 'w') as txt:
        txt.writelines(all_label)
        txt.close()
    return True

def WriteLabel_trn_tst(rootdirv, proportion=0.5, total=None):
    '''
    Write trn and tst label for all images in rootdirv
    :param rootdirv: the images' root directory path
    :param proportion: indicate the trn:tst(by number)
    :param total: the total number of each category, if you want align size, set this parameter
    :return: True of False
    '''
    if not os.path.exists(rootdirv):
        return False
    trn_label = []
    tst_label = []
    num_label = {}
    for parent, dirnames, filenames in os.walk(rootdirv):
        labelId = parent.replace(rootdirv, '').replace('\\', '/').lstrip('/').split('/')[0]
        if not labelId.isnumeric():
            continue
        num_label[labelId] = 0
        mix_label = []
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext != '.jpg' and ext != '.JPG':
                continue
            sourceFile = os.path.join(parent, filename)
            if not os.path.exists(sourceFile):
                continue
            label = sourceFile.replace(rootdirv, '').replace('\\', '/').lstrip('/') + ' ' + labelId + '\n'
            mix_label.append(label)
        np.random.shuffle(mix_label)
        # align size, if total is not None
        if total and total < len(mix_label) + num_label[labelId]:
            assert num_label[labelId] <= total
            mix_label = mix_label[:total - num_label[labelId]]
        num_label[labelId] += len(mix_label)
        index = int(len(mix_label)*proportion)
        trn_label.extend(mix_label[:index])
        tst_label.extend(mix_label[index:])
    print(num_label)
    with open(rootdirv + r'\trn_label.txt', 'w') as trn_txt:
        trn_txt.writelines(trn_label)
        trn_txt.close()
    with open(rootdirv + r'\tst_label.txt', 'w') as tst_txt:
        tst_txt.writelines(tst_label)
        tst_txt.close()
    return True
"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import numpy as np
import os
from PIL import Image

__all__ = ['SegmentationMetric']

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixel_accuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def class_pixel_accuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def mean_pixel_accuracy(self):
        classAcc = self.class_pixel_accuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def mean_intersection_over_union(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def gen_confusion_matrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def frequency_weighted_intersection_over_union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def add_batch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.gen_confusion_matrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


def pre_process(label, mask):
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] != 0:
                label[i][j] = 1
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] != 0:
                mask[i][j] = 1


def batch_process(label_dir, mask_dir):
    avg_mIoU = 0.0
    avg_pa = 0.0
    cnt = 0.0

    for f in os.listdir(mask_dir):  # mask
        label = np.array(Image.open(os.path.join(label_dir, f)))
        mask = np.array(Image.open(os.path.join(mask_dir, f)))
        pre_process(label, mask)
        metric = SegmentationMetric(2)  # 3表示有3个分类，有几个分类就填几
        metric.add_batch(mask, label)
        pa = metric.pixel_accuracy()
        cpa = metric.class_pixel_accuracy()
        mpa = metric.mean_pixel_accuracy()
        mIoU = metric.mean_intersection_over_union()
        avg_mIoU = avg_mIoU + mIoU
        avg_pa = avg_pa + pa
        cnt = cnt + 1
        print("------------")
        print("current img is:{}".format(f))
        print('pa is : %f' % pa)
        print('cpa is :')  # 列表
        print(cpa)
        print('mpa is : %f' % mpa)
        print('mIoU is : %f' % mIoU)

    print("avg_mIoU is :%f" % (avg_mIoU / cnt))
    print("avg_pa is :%f" % (avg_pa / cnt))


if __name__ == '__main__':
    batch_process("D:\\zl\\GraduationThesis\\data\\new_data\\label",
                  "D:\\zl\\GraduationThesis\\material\\test\\test-1_result_2021_01_25")
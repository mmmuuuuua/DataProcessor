import random
import os


def process(input, train_file, val_file):
    all_samples = os.listdir(input)
    test_num = len(all_samples) // 20
    print("the total number of samples is {}, of which the training sample is {} and the test sample is{}".
          format(len(all_samples), len(all_samples) - test_num, test_num))
    random.shuffle(all_samples)
    train_samples = all_samples[:test_num]
    test_samples = all_samples[test_num:]
    with open(train_file, "w") as f:
        for train_sample in train_samples:
            f.writelines(train_sample + "\n")
    with open(val_file, "w") as f:
        for test_sample in test_samples:
            f.writelines(test_sample + "\n")


if __name__ == '__main__':
    input = 'D:\\learning\\GraduationThesis\\data\\ROCKtrainval\\SequenceImages'
    train_file = 'D:\\learning\\GraduationThesis\\data\\ROCKtrainval\\ImageSets\\Segmentation\\train.txt'
    val_file = 'D:\\learning\\GraduationThesis\\data\\ROCKtrainval\\ImageSets\\Segmentation\\val.txt'
    process(input, train_file, val_file)

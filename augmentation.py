#!/usr/bin/python
# -*- coding: UTF-8 -*-
import imgaug as ia
from imgaug import augmenters as iaa
import time
import cv2
import os
import numpy as np

from ImageFileIO import *

from skimage import exposure
import random


def img_enbright(images, random_state, parents, hooks):
    for index in range(len(images)):
        images[index] = exposure.adjust_gamma(images[index], 0.8)
    return images


def img_endark(images, random_state, parents, hooks):
    for index in range(len(images)):
        images[index] = exposure.adjust_gamma(images[index], random.uniform(1.1, 3.0))
    return images


def img_endark_level1(images, random_state, parents, hooks):
    for index in range(len(images)):
        images[index] = exposure.adjust_gamma(images[index], random.uniform(1.1, 1.4))
    return images


def img_endark_level2(images, random_state, parents, hooks):
    for index in range(len(images)):
        images[index] = exposure.adjust_gamma(images[index], random.uniform(1.4, 1.7))
    return images


def img_endark_level3(images, random_state, parents, hooks):
    for index in range(len(images)):
        images[index] = exposure.adjust_gamma(images[index], random.uniform(1.7, 2.0))
    return images


def img_endark_level4(images, random_state, parents, hooks):
    for index in range(len(images)):
        images[index] = exposure.adjust_gamma(images[index], random.uniform(2.0, 2.5))
    return images


def img_endark_level5(images, random_state, parents, hooks):
    for index in range(len(images)):
        images[index] = exposure.adjust_gamma(images[index], random.uniform(2.5, 3))
    return images


def keypoint_func(keypoints_on_images, random_state, parents, hooks):
    return keypoints_on_images


# only the sequential iamges are supported
EnBrightOrEnDark = iaa.OneOf([
    iaa.Lambda(img_enbright, keypoint_func),
    iaa.Lambda(img_enbright, keypoint_func),
    # iaa.Lambda(img_endark, keypoint_func),
    # iaa.Add(-50, per_channel=True)
])

EnDark = iaa.OneOf([
    iaa.Lambda(img_endark, keypoint_func),
    iaa.Lambda(img_endark_level1, keypoint_func),
    iaa.Lambda(img_endark_level2, keypoint_func),
    # iaa.Lambda(img_endark_level3, keypoint_func),
    # iaa.Lambda(img_endark_level4, keypoint_func),
    # iaa.Lambda(img_endark_level5, keypoint_func)
])

EnDark_0 = iaa.OneOf([
    iaa.Lambda(img_endark, keypoint_func),
    # iaa.Lambda(img_endark_level1, keypoint_func),
    # iaa.Lambda(img_endark_level2, keypoint_func),
    iaa.Lambda(img_endark_level3, keypoint_func),
    iaa.Lambda(img_endark_level4, keypoint_func),
    iaa.Lambda(img_endark_level5, keypoint_func)
])

EnDark_1 = iaa.OneOf([
    iaa.Lambda(img_endark_level1, keypoint_func),
    iaa.Lambda(img_endark_level2, keypoint_func),
    iaa.Lambda(img_endark_level3, keypoint_func),
    # iaa.Lambda(img_endark_level4, keypoint_func),
    # iaa.Lambda(img_endark_level5, keypoint_func)
])

EnDark_2 = iaa.OneOf([
    iaa.Lambda(img_endark_level1, keypoint_func),
    # iaa.Lambda(img_endark_level2, keypoint_func),
    # iaa.Lambda(img_endark_level3, keypoint_func),
    # iaa.Lambda(img_endark_level4, keypoint_func),
    # iaa.Lambda(img_endark_level5, keypoint_func)
])


seq1 = iaa.Sequential([
    iaa.Crop(px=(0, 25)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order


ia.seed(int(time.time()))
seq2 = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image. It maight change the color.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order

seq3 = iaa.Sequential([
    # Normalize contrast by a factor of 0.5 to 1.5, sampled randomly per image. It maight change the color.
    iaa.ContrastNormalization((1.0, 2.5)),
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(1.0, 2.5)) # sigma=(0, 2.5)
    ),
    # Augmenter that sharpens images and overlays the result with the original image.
    iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.5, 1.2)),
    # Multiply values of pixels with possibly different values for neighbouring pixels, making each pixel darker or brighter.
    # Multiply each pixel with a random value between 0.8 and 1.1:
    iaa.MultiplyElementwise((0.8,1.1)),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.7, 2.0)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-90, 90),
        shear=(-16, 16),
        order=1
        # mode=ia.ALL,
        # cval=(0,255)
    ),
    # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Crop(px=(0, 25)),
    # Add gaussian noise. For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND channel. This can change the color (not only brightness) of the pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    iaa.PiecewiseAffine(scale=(0.01,0.05)),
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
], random_order=True) # apply augmenters in random order

seq4 = iaa.Sequential([
    # Normalize contrast by a factor of 0.5 to 1.5, sampled randomly per image. It maight change the color.
    ## iaa.ContrastNormalization((1.0, 2.5)),
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(1.0, 2.5))),
    # Augmenter that sharpens images and overlays the result with the original image.
    iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.5, 1.2)),
    # Drop 0 to 5% of all pixels by converting them to black pixels, but do that on
    # a lower-resolution version of the image that has 5% to 50% of the original size, leading to large rectangular areas being dropped:
    iaa.Sometimes(0.1, iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.8))),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.7, 2.0)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-90, 90),
        shear=(-16, 16),
        order=1
        # mode=ia.ALL,
        # cval=(0,255)
    ),
    # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Crop(px=(0, 25)),
    # Add gaussian noise. For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND channel. This can change the color (not only brightness) of the pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    iaa.PiecewiseAffine(scale=(0.01,0.05)),
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.Sometimes(0.15, iaa.ElasticTransformation(alpha=(0, 3.0), sigma=0.25))
], random_order=True) # apply augmenters in random order


# For PLA
seq4_pla = iaa.Sequential([
    # Normalize contrast by a factor of 0.5 to 1.5, sampled randomly per image. It maight change the color.
    ## iaa.ContrastNormalization((1.0, 2.5)),
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(1.0, 2.0))),
    # Augmenter that sharpens images and overlays the result with the original image.
    iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.5, 1.2)),
    # Drop 0 to 5% of all pixels by converting them to black pixels, but do that on
    # a lower-resolution version of the image that has 5% to 50% of the original size, leading to large rectangular areas being dropped:
    iaa.Sometimes(0.1, iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.9))),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.7, 2.0)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-90, 90),
        shear=(-16, 16),
        order=1
        # mode=ia.ALL,
        # cval=(0,255)
    ),
    # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Crop(px=(0, 25)),
    # Add gaussian noise. For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND channel. This can change the color (not only brightness) of the pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    ## iaa.PiecewiseAffine(scale=(0.01,0.05)),
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.Sometimes(0.15, iaa.ElasticTransformation(alpha=(0, 3.0), sigma=0.25))
], random_order=True) # apply augmenters in random order

# For Original(just simply transfrom)
seq_ori = iaa.Sequential([
    # endark the image
    # iaa.Lambda(img_enbright, keypoint_func),
    EnDark_2,
    # Small gaussian blur with random sigma between 0 and 0.5, But we only blur about 50% of all images.
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(1.0, 1.5))),
    # Augmenter that sharpens images and overlays the result with the original image.
    iaa.Sharpen(alpha=(0.1, 0.5), lightness=(0.5, 1.0)),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(scale={"x": (0.3, 1.2), "y": (0.3, 1.2)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-90, 90),
        order=1
    ),
    iaa.Fliplr(0.5), # horizontally flip 50% of the images

], random_order=True) # apply augmenters in random order


CBLN = iaa.Sequential([
    # Normalize contrast by a factor of 0.5 to 1.5, sampled randomly per image. It maight change the color.
    iaa.ContrastNormalization((0.9, 1.2)),
    # Small gaussian blur with random sigma between 0 and 0.5, But we only blur about 50% of all images.
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(1.0, 1.5))),
    # endark the image
    EnBrightOrEnDark,
    # Add SaltAndPepper noise. For 50% of all images, we sample the noise once per pixel.
    iaa.Pepper((0.01, 0.02)),
    # Drop 0 to 5% of all pixels by converting them to black pixels, but do that on
    # a lower-resolution version of the image that has 5% to 50% of the original size, leading to large rectangular areas being dropped:
    iaa.CoarseDropout((0.0, 0.02), size_percent=(0.1, 0.35)),
], random_order=True) # apply augmenters in random order

CBLN1 = iaa.Sequential([
    # Normalize contrast by a factor of 0.5 to 1.5, sampled randomly per image. It maight change the color.
    iaa.ContrastNormalization(0.9),
    # Small gaussian blur with random sigma between 0 and 0.5, But we only blur about 50% of all images.
    iaa.GaussianBlur(1.0),
    # endark the image
    EnBrightOrEnDark,
    # Add SaltAndPepper noise. For 50% of all images, we sample the noise once per pixel.
    iaa.Pepper(0.01),
    # Drop 0 to 5% of all pixels by converting them to black pixels, but do that on
    # a lower-resolution version of the image that has 5% to 50% of the original size, leading to large rectangular areas being dropped:
    iaa.CoarseDropout((0.0, 0.02), size_percent=(0.1, 0.35)),
], random_order=True) # apply augmenters in random order

CBLN2 = iaa.Sequential([
    # Normalize contrast by a factor of 0.5 to 1.5, sampled randomly per image. It maight change the color.
    iaa.ContrastNormalization(1.1),
    # Small gaussian blur with random sigma between 0 and 0.5, But we only blur about 50% of all images.
    # iaa.GaussianBlur(sigma=1.5),
    # endark the image
    EnBrightOrEnDark,
    # Add SaltAndPepper noise. For 50% of all images, we sample the noise once per pixel.
    iaa.Pepper(0.02),
    # Drop 0 to 5% of all pixels by converting them to black pixels, but do that on
    # a lower-resolution version of the image that has 5% to 50% of the original size, leading to large rectangular areas being dropped:
    iaa.CoarseDropout((0.0, 0.02), size_percent=(0.1, 0.35)),
], random_order=True) # apply augmenters in random order

xingbian = iaa.Sequential([
    # iaa.PiecewiseAffine(scale=(0.03,0.08)),
    iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.7, 2.0)},
               # translate_percent={"x": (-0.2, 0.2), "y": (-0.1, 0.1)},
               rotate=(-90, 90),
               shear=(-32, 32),
               order=1
               # mode=ia.ALL,
               # cval=(0,255)
               ),
    # iaa.Scale({"height": (0.5, 1), "width": (0.75, 1)}, interpolation=["linear", "cubic"])
], random_order=True)

xingbian1 = iaa.Sequential([
    # iaa.PiecewiseAffine(scale=(0.03,0.08)),
    iaa.Affine(scale={"x": 0.5, "y": 1.5},
               # translate_percent={"x": -0.2, "y": -0.1},
               # rotate=-90,
               shear=-15,
               order=1
               # mode=ia.ALL,
               # cval=(0,255)
               ),
    # iaa.Scale({"height": (0.5, 1), "width": (0.75, 1)}, interpolation=["linear", "cubic"])
], random_order=True)

xingbian2 = iaa.Sequential([
    # iaa.PiecewiseAffine(scale=(0.03,0.08)),
    iaa.Affine(scale={"x": 0.7, "y": 1.3},
               # translate_percent={"x": -0.2, "y": -0.1},
               # rotate=-90,
               shear=-15,
               order=1
               # mode=ia.ALL,
               # cval=(0,255)
               ),
    # iaa.Scale({"height": (0.5, 1), "width": (0.75, 1)}, interpolation=["linear", "cubic"])
], random_order=True)

xingbian3 = iaa.Sequential([
    # iaa.PiecewiseAffine(scale=(0.03,0.08)),
    iaa.Affine(scale={"x": 0.9, "y": 1.1},
               # translate_percent={"x": -0.2, "y": -0.1},
               # rotate=-90,
               shear=-30,
               order=1
               # mode=ia.ALL,
               # cval=(0,255)
               ),
    # iaa.Scale({"height": (0.5, 1), "width": (0.75, 1)}, interpolation=["linear", "cubic"])
], random_order=True)

xingbian4 = iaa.Sequential([
    # iaa.PiecewiseAffine(scale=(0.03,0.08)),
    iaa.Affine(scale={"x": 0.5, "y": 1.5},
               # translate_percent={"x": -0.2, "y": -0.1},
               # rotate=-90,
               shear=-30,
               order=1
               # mode=ia.ALL,
               # cval=(0,255)
               ),
    # iaa.Scale({"height": (0.5, 1), "width": (0.75, 1)}, interpolation=["linear", "cubic"])
], random_order=True)

xingbian5 = iaa.Sequential([
    # iaa.PiecewiseAffine(scale=(0.03,0.08)),
    iaa.Affine(scale={"x": 0.7, "y": 1.3},
               # translate_percent={"x": -0.2, "y": -0.1},
               # rotate=-90,
               shear=-30,
               order=1
               # mode=ia.ALL,
               # cval=(0,255)
               ),
    # iaa.Scale({"height": (0.5, 1), "width": (0.75, 1)}, interpolation=["linear", "cubic"])
], random_order=True)

xingbian6 = iaa.Sequential([
    # iaa.PiecewiseAffine(scale=(0.03,0.08)),
    iaa.Affine(scale={"x": 0.9, "y": 1.1},
               # translate_percent={"x": -0.2, "y": -0.1},
               # rotate=-90,
               shear=-30,
               order=1
               # mode=ia.ALL,
               # cval=(0,255)
               ),
    # iaa.Scale({"height": (0.5, 1), "width": (0.75, 1)}, interpolation=["linear", "cubic"])
], random_order=True)

xingbianCBLN1 = iaa.Sequential([
    # iaa.PiecewiseAffine(scale=(0.03,0.08)),
    iaa.Affine(scale={"x": 0.5, "y": 1.5},
               # translate_percent={"x": -0.2, "y": -0.1},
               # rotate=-90,
               shear=-15,
               order=1
               # mode=ia.ALL,
               # cval=(0,255)
               ),
    CBLN2
    # iaa.Scale({"height": (0.5, 1), "width": (0.75, 1)}, interpolation=["linear", "cubic"])
], random_order=True)

xingbianCBLN2 = iaa.Sequential([
    # iaa.PiecewiseAffine(scale=(0.03,0.08)),
    iaa.Affine(scale={"x": 0.7, "y": 1.3},
               # translate_percent={"x": -0.2, "y": -0.1},
               # rotate=-90,
               shear=-15,
               order=1
               # mode=ia.ALL,
               # cval=(0,255)
               ),
    CBLN2
    # iaa.Scale({"height": (0.5, 1), "width": (0.75, 1)}, interpolation=["linear", "cubic"])
], random_order=True)

xingbianCBLN3 = iaa.Sequential([
    # iaa.PiecewiseAffine(scale=(0.03,0.08)),
    iaa.Affine(scale={"x": 0.9, "y": 1.1},
               # translate_percent={"x": -0.2, "y": -0.1},
               # rotate=-90,
               shear=-15,
               order=1
               # mode=ia.ALL,
               # cval=(0,255)
               ),
    CBLN2
    # iaa.Scale({"height": (0.5, 1), "width": (0.75, 1)}, interpolation=["linear", "cubic"])
], random_order=True)

xingbianCBLN4 = iaa.Sequential([
    # iaa.PiecewiseAffine(scale=(0.03,0.08)),
    iaa.Affine(scale={"x": 0.5, "y": 1.5},
               # translate_percent={"x": -0.2, "y": -0.1},
               # rotate=-90,
               shear=-30,
               order=1
               # mode=ia.ALL,
               # cval=(0,255)
               ),
    CBLN2
    # iaa.Scale({"height": (0.5, 1), "width": (0.75, 1)}, interpolation=["linear", "cubic"])
], random_order=True)

xingbianCBLN5 = iaa.Sequential([
    # iaa.PiecewiseAffine(scale=(0.03,0.08)),
    iaa.Affine(scale={"x": 0.7, "y": 1.3},
               # translate_percent={"x": -0.2, "y": -0.1},
               # rotate=-90,
               shear=-30,
               order=1
               # mode=ia.ALL,
               # cval=(0,255)
               ),
    CBLN2
    # iaa.Scale({"height": (0.5, 1), "width": (0.75, 1)}, interpolation=["linear", "cubic"])
], random_order=True)

xingbianCBLN6 = iaa.Sequential([
    # iaa.PiecewiseAffine(scale=(0.03,0.08)),
    iaa.Affine(scale={"x": 0.9, "y": 1.1},
               # translate_percent={"x": -0.2, "y": -0.1},
               # rotate=-90,
               shear=-30,
               order=1
               # mode=ia.ALL,
               # cval=(0,255)
               ),
    CBLN2
    # iaa.Scale({"height": (0.5, 1), "width": (0.75, 1)}, interpolation=["linear", "cubic"])
], random_order=True)

fangshe = iaa.Sequential([
    iaa.PiecewiseAffine(scale=(0.03,0.08)),
    # iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.7, 2.0)},
    #            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
    #            rotate=(-90, 90),
    #            shear=(-16, 16),
    #            order=1
    #            # mode=ia.ALL,
    #            # cval=(0,255)
    #            ),
    # iaa.Scale({"height": (0.5, 1), "width": (0.75, 1)}, interpolation=["linear", "cubic"])
], random_order=True)

fangshe1 = iaa.Sequential([
    iaa.PiecewiseAffine(scale=0.05),
    # iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.7, 2.0)},
    #            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
    #            rotate=(-90, 90),
    #            shear=(-16, 16),
    #            order=1
    #            # mode=ia.ALL,
    #            # cval=(0,255)
    #            ),
    # iaa.Scale({"height": (0.5, 1), "width": (0.75, 1)}, interpolation=["linear", "cubic"])
], random_order=True)

fangshe2 = iaa.Sequential([
    iaa.PiecewiseAffine(scale=0.1),
    # iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.7, 2.0)},
    #            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
    #            rotate=(-90, 90),
    #            shear=(-16, 16),
    #            order=1
    #            # mode=ia.ALL,
    #            # cval=(0,255)
    #            ),
    # iaa.Scale({"height": (0.5, 1), "width": (0.75, 1)}, interpolation=["linear", "cubic"])
], random_order=True)


'''
# use this code to test a single image
image = cv2.imread(r'D:\WorkSpace\DeepLearning\PLGrainDataBase\original\2\PLA00001_OBJECT.jpg')
aug = iaa.Add(50, per_channel=True)
image_aug = aug.augment_image(image)
cv2.imshow("src", image)
cv2.imshow("rst", image_aug)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


'''
# use this code to test the sequential images
image = cv2.imread(r'D:\WorkSpace\DeepLearning\PLGrainDataBase\3_5\0\QUA00004_OBJECT.jpg')
# image = cv2.imread(r'D:\WorkSpace\DeepLearning\PLGrainDataBase\original\0\QUA00315_OBJECT.jpg')
images = []
for i in range(10):
    images.append(image)
images_aug = xingbian.augment_images(images)
cv2.imshow("src", image)
for i in range(0, len(images_aug)):
    cv2.imshow("rst_"+str(i), images_aug[i])
    cv2.imwrite(r"D:\WorkSpace\DeepLearning\PLGrainDataBase\3_5\0\CBLN_rst_"+str(i)+".jpg", images_aug[i])
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
# use this code for imgs_imgaug by label.txt
labelfile = r"D:\WorkSpace\DeepLearning\PLGrainDataBase\aug4_label.txt"
[images, dir_path, images_filepath] = ReadImagesFromLabelFile(labelfile, ' ')

images_aug = seq_ori.augment_images(images)

WriteImagesFromRelativePathList(images_aug, dir_path + "tmp/", images_filepath)
'''

'''
# use this code for imgs_imgaug by root directory
rootdir = r'D:\WorkSpace\DeepLearning\PLGrainDataBase\aug4\3'
[images, images_filepath] = ReadImagesFromDirectory(rootdir)

# for i in range(0, 7):
#     images_dst = seq4_pla.augment_images(images)
#     images_filepath_dst = list(map(lambda pth: pth.replace('original', 'test_seq4_'+str(i+1)), images_filepath))
#     WriteImagesFromPathList(images_dst, images_filepath_dst)

images_dst = seq_ori.augment_images(images)
images_filepath_dst = list(map(lambda str: str.replace('aug4', 'test_seq4_7'), images_filepath))
WriteImagesFromPathList(images_dst, images_filepath_dst)
'''


'''
rootdir = r'D:\WorkSpace\DeepLearning\PLGrainDataBase\aug4_aug'
WriteLabelEx(rootdir, os.path.dirname(rootdir), True, numEveryone={'0':10000, '1':10000, '2':10000, '3':10000})
# WriteLabel(rootdir, os.path.dirname(rootdir), False)
'''

# rootdir = r'D:\WorkSpace\Microsoft Visual Studio 14.0\Projects\TestRecoRate\TestRecoRate\MIresult_3'
# WriteLabel(rootdir, os.path.dirname(rootdir), False)


def sequence_image_augmentation(input_path, image_augmentation_methods, start_index):
    all_dirs = [os.path.join(input_path, img_dir) for img_dir in os.listdir(input_path)]
    for d in all_dirs:
        for method in image_augmentation_methods:
            output_dir = os.path.join(input_path, "QUA" + str(start_index))
            os.mkdir(output_dir)
            start_index = start_index + 1
            print(start_index)
            for img in os.listdir(d):
                image = cv2.imread(os.path.join(d, img))
                image_aug = method.augment_image(image)
                cv2.imwrite(os.path.join(output_dir, img), image_aug)


if __name__ == '__main__':
    # img_path = 'D:\\zl\\GraduationThesis\\data\\classification_data\\classification_data\\ALK\\ALK7\\C6.jpg'
    # image = cv2.imread(img_path)
    # images_aug = xingbian3.augment_image(image)
    # cv2.imshow("src", image)
    #
    # cv2.imshow("rst_" + "1", images_aug)
    # # cv2.imwrite("D:\\zl\\GraduationThesis\\data\\classification_data\\classification_data\\PLA\\tmp" + str(6) + ".jpg", images_aug[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    input_path = "D:\\zl\\GraduationThesis\\data\\classification_data\\classification_data\\QUA"
    # image_augmentation_methods = [xingbian1, xingbian2, xingbian3, xingbian4, xingbian5, xingbian6,
    #                               xingbianCBLN1, xingbianCBLN2, xingbianCBLN3,
    #                               xingbianCBLN4, xingbianCBLN5, xingbianCBLN6]
    # image_augmentation_methods = [xingbian2, xingbian3, xingbian6]
    image_augmentation_methods = [xingbian3]
    start_index = 1526
    sequence_image_augmentation(input_path, image_augmentation_methods, start_index)
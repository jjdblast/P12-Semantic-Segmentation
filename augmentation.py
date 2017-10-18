import cv2
import numpy as np


def rotate(image,
           angle, tx, ty, scale,
           interpolation=cv2.INTER_LINEAR,
           border_value=0):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
    M[0, 2] = tx
    M[1, 2] = ty
    image = cv2.warpAffine(image, M, (w, h),
                           flags=interpolation, borderValue=border_value)
    return image


def rotate_both(image, label, p=0.5, ignore_label=1):
    if np.random.rand() > p:
        return image, label
    h, w = image.shape[:2]
    angle = np.random.uniform(-10, 10)
    tx = np.random.uniform(-w // 128, w // 128)
    ty = np.random.uniform(-h // 128, h // 128)
    scale = np.random.uniform(0.98, 1.02)
    image = rotate(image, angle, tx, ty, scale, border_value=(127, 127, 127))
    label = rotate(label, angle, tx, ty, scale,
                   interpolation=cv2.INTER_NEAREST, border_value=ignore_label)
    return image, label


def flip_both(image, label, p=0.5):
    if np.random.rand() > p:
        return image, label
    image = np.fliplr(image)
    label = np.fliplr(label)
    return image, label


def blur(image, p=0.5):
    if np.random.rand() > p:
        return image
    k = np.random.choice([3, 5, 7])
    image = cv2.GaussianBlur(image, (k, k), 0)
    return image


def blur_both(image, label, p=0.5):
    image = blur(image)
    return image, label

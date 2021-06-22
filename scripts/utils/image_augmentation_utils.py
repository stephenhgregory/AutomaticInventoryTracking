"""Contains functions to augment images"""

from typing import Tuple
from utils import logging_utils
import cv2
import numpy as np


def preprocess_labels(label_images: np.ndarray) -> np.ndarray:
    """
    Simply calls preprocess_label on all labels in label_imgs

    Parameters
    ----------
    label_images: numpy array of images

    Returns
    -------
    uint8 numpy array (preprocessed images)
    """
    for i in range(label_images.shape[0]):
        label_images[i] = preprocess_label(label_images[i])

    return label_images


@logging_utils.time_logger
def preprocess_label(label_img: np.ndarray) -> np.ndarray:
    """
    Function to organize which preprocessing functions to call on label image

    Parameters
    ----------
    label_img: uint8 numpy array of pixel values

    Returns
    -------
    uint8 numpy array (preprocessed image)
    """
    # these 5 are the normal chain of preprocessing
    # grayscale -> blur -> thresholding -> erosion -> dilation 
    # (erosion and dilation reversed rather than inverting image before and after)
    gray_label = gray(label_img)
    blurred_label = blur(gray_label, kernel_size=(7, 7))

    label = threshold(blurred_label, algorithm='gaussian')
    label = dilate(label)
    label = erode(label)

    # these are additional steps to improve output
    label = blur(label, kernel_size=(5, 5))
    label = dilate(label)
    label = blur(label, kernel_size=(3, 3))

    # convert label back to rgb from grayscale
    label = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)

    ''' Just logging
    image_utils.show_images(np.array([gray_label, blurred_label, label]),
                            ["gray_label", "blurred_label", "final_label"])
    '''

    return label


def apply_CLAHE(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Performs Contrast Limited Adaptive Histogram Equalization on an input image

    :param image: An numpy array representing an image to be equalized
    :type image: Numpy array
    :param clip_limit: The contrast limit of any given tile in the transformation
    :type clip_limit: float
    :param tile_grid_size: The size of each sub-patch that gets normalized
    :type tile_grid_size: tuple of ints

    :return: An augmented image
    :rtype: Numpy array
    """

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Return the augmented image
    return clahe.apply(image)


def gray(img):
    """
    Function to gray scale image, takes in image and returns the image in grayscale

    Parameters
    ----------
    img: uint8 np array of pixel values

    Returns
    -------
    uint8 np array (grayscaled image)
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def blur(img, kernel_size: Tuple[int, int] = (5, 5)):
    """
    Function to blur image, takes in image and returns the blurred image

    Parameters
    ----------
    img: uint8 np array of pixel values
    kernel_size: Kernel size uesd to apply Gaussian Smoothing

    Returns
    -------
    uint8 np array (the blurred image)
    """
    img_blur = cv2.GaussianBlur(img, kernel_size, 0)
    return img_blur


def threshold(img: np.ndarray, threshold_value: int = 100, algorithm: str = 'gaussian') -> np.ndarray:
    """
    Function to threshold image, takes in image and returns the thresholded image

    Parameters
    ----------
    img: uint8 numpy array of pixel values
    threshold_value: The pixel value threshold used to split black and white binarization
    algorithm: The particular algorithm used to binarize the image

    Returns
    --------
    uint8 numpy array (the blurred image)
    """

    if algorithm == 'gaussian':
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif algorithm == 'otsu':
        img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]
    return img


# function to blur image, takes in image and returns the blurred image
def erode(img: np.ndarray) -> np.ndarray:
    """
    Function to use erosion on image, takes in image and returns the eroded image

    Parameters
    ----------
    img: uint8 np array of pixel values

    Returns
    -------
    uint8 np array (the blurred image)
    """
    kernel = np.ones((3, 3), np.uint8)
    img_erode = cv2.erode(img, kernel, iterations=1)
    return img_erode


# function to blur image, takes in image and returns the blurred image
def dilate(img):
    """
    Function to use dilation on image, takes in image and returns the dilated image

    Parameters
    -----------
    uint8 np array of pixel values

    Returns
    -------
    uint8 np array (the blurred image)
    """
    kernel = np.ones((2, 2), np.uint8)
    img_dilate = cv2.dilate(img, kernel, iterations=1)
    return img_dilate

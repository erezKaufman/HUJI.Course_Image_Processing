import numpy as np

import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy.misc import imread, comb
from scipy import signal
PIXEL_MAX_INTENSITY = 255

def read_image(filename, representation):
    """
    The function will return an image file represented as the user wished it to be
    :param filename:  a string that tells the image location
    :param representation:  an integer representing in what way we wish the image to appear.
    1 - gray scale
    2 - rgb
    :return: a numpy image format
    """
    newIm = imread(filename)
    if representation == 2 or len(newIm.shape) == 2:
        floatIm = newIm.astype(np.float64)
        floatIm /= PIXEL_MAX_INTENSITY
        return floatIm
    else:
        return rgb2gray(newIm)

def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Task 3.1 a
    should return pyr as std array (NOT NUMPY!) with max len of max_levels, and each element of the array is a
    grayscale image.
    the filter_vec is an output which is 1D-row of size filter_size used for the pyramid construction.
    :param im: a grayscale image with double values in [0, 1] (e.g. the output of ex1’s read_image with the
               representation set to 1)
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
                        in constructing the pyramid filter (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25]).
    :return: tuple of two values - pyr, filter_vec
    """
    pass


def  build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Task 3.1 b
    should return pyr as std array (NOT NUMPY!) with max len of max_levels, and each element of the array is a
    grayscale image.
    the filter_vec is an output which is 1D-row of size filter_size used for the pyramid construction.

    :param im: a grayscale image with double values in [0, 1] (e.g. the output of ex1’s read_image with the
               representation set to 1)
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
                        in constructing the pyramid filter (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25]).
    :return: tuple of two values - pyr, filter_vec
    """
    pass



def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    Task 3.2
    both lpyr and filter_vec are garuenteed from task 3.1
    :param lpyr: Laplacian pyramid
    :param filter_vec: filter
    :param coeff:
    :return: img
    """
    pass


def render_pyramid(pyr, levels):
    """

    :param pyr:
    :param levels:
    :return:
    """
    pass


def display_pyramid(pyr, levels):
    """

    :param pyr:
    :param levels:
    :return: None
    """
    pass


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """

    :param im1:             grayscale image to be blended.
    :param im2:             grayscale image to be blended.
    :param mask:            a boolean (i.e. dtype == np.bool) mask containing True and False representing which parts
                            of im1 and im2 should appear in the resulting im_blend. Note that a value of True
                            corresponds to 1,
                            and False corresponds to 0.
    :param max_levels:      the max_levels parameter you should use when generating the Gaussian and Laplacian
                            pyramids.
    :param filter_size_im:  the size of the Gaussian filter (an odd scalar that represents a squared filter) which
                            defining the filter used in the construction of the Laplacian pyramids of im1 and im2.
    :param filter_size_mask:the size of the Gaussian filter(an odd scalar that represents a squared filter) which
                            defining the filter used in the construction of the Gaussian pyramid of mask.
    :return:  im_blend
    """
    pass



def blending_example1():
    """

    :return: im1, im2, mask, im_blend
    """
    pass

def blending_example2():
    """

    :return: im1, im2, mask, im_blend
    """
    pass
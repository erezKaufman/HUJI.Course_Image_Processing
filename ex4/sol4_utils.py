
import os
import numpy as np
from scipy.ndimage.filters import convolve as convolve
from skimage.color import rgb2gray
from scipy.misc import imread
from scipy import signal

PIXEL_MAX_INTENSITY = 255

#################### help functions for SOL 4 #####################

def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Task 3.1 b
    should return pyr as std array (NOT NUMPY!) with max len of max_levels, and each element of the array is a
    grayscale image.
    the filter_vec is an output which is 1D-row of size filter_size used for the pyramid construction.

    :param im:           a grayscale image with double values in [0, 1] (e.g. the output of ex1’s read_image with the
                         representation set to 1)
    :param max_levels:   the maximal number of levels in the resulting pyramid.
    :param filter_size:  the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
                         in constructing the pyramid filter (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25]).
    :return:             tuple of two values - pyr, filter_vec
    """
    gaus_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    pyr = []

    filter_vec *= 2

    # filter_vec = filter_vec.reshape(filter_size, 1)
    filter_vec_t = filter_vec.transpose()
    level = 0
    for level in range(len(gaus_pyr) - 1):
        expand_image = expand(gaus_pyr[level + 1], filter_vec, filter_vec_t)
        new_image = gaus_pyr[level] - expand_image
        pyr.append(new_image)
    pyr.append(gaus_pyr[level + 1])

    return pyr, filter_vec

def expand(im, filter_vec, filter_vec_t):
    new_size = np.array(im.shape) * 2
    zero_matrix = np.zeros(new_size, dtype=np.float64)
    zero_matrix[::2, ::2] = im
    im_blured = convolve(zero_matrix, filter_vec, )
    im_blured = convolve(im_blured, filter_vec_t)
    return im_blured[::, ::]

def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    Task 3.2
    both lpyr and filter_vec are garuenteed from task 3.1
    :param lpyr:       Laplacian pyramid
    :param filter_vec: filter
    :param coeff:
    :return: img
    """

    new_image = lpyr[0]
    temp_image = expand(lpyr[len(lpyr) - 1] * coeff[len(coeff) - 1], filter_vec, filter_vec.transpose())
    for index in range(len(lpyr) - 2, -1, -1):
        new_image = (lpyr[index]) + temp_image

        temp_image = expand(new_image, filter_vec, filter_vec.transpose())

    return new_image

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    Task 4
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
    L1, filter_vec1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L2, filter_vec2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)

    Gm, filter_vec3 = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)
    l_out = []
    for k in range(len(L2)):
        first_product = (Gm[k]) * (L1[k])
        second_product = (1 - Gm[k]) * L2[k]
        result = first_product + second_product

        l_out.append(result)
    returned_image = laplacian_to_image(l_out, filter_vec1, [1 for i in range(max_levels)]).clip(0, 1)
    return returned_image


def getGaussianKernel(size):
    """
    returns a gaussian kernel of a given size
    :param size: the size of the kernal
    :return: a gaussian kernel of size "size"
    """
    start_vector = np.array([1, 1])
    bin_vector = np.array([1, 1])
    for i in range(size - 2):
        bin_vector = np.convolve(start_vector, bin_vector)

    guass_matrix = np.outer(np.array(bin_vector), np.array(bin_vector))
    return guass_matrix / np.sum(guass_matrix)


def blur_spatial(im, kernel_size):
    """
    blurs an image with a guassian kernel
    :param im: the image to blur
    :param kernel_size: the kernel size
    :return: a blurred image
    """
    if (kernel_size == 1):
        return im
    guass_kernel = getGaussianKernel(kernel_size)
    return signal.convolve2d(im, guass_kernel, mode='same', boundary='fill', fillvalue=0)


def reduce(im, gaus_filter, gaus_filter_t):
    im_blured = convolve(im, gaus_filter)
    im_blured = convolve(im_blured, gaus_filter_t)
    return im_blured[::2, ::2]


def read_image(filename, representation):
    """
    The function will return an image file represented as the user wished it to be
    :param filename:        a string that tells the image location
    :param representation:  an integer representing in what way we wish the image to appear.
                            1 - gray scale
                            2 - rgb
    :return:                a numpy image format
    """
    newIm = imread(filename)
    if representation == 2 or len(newIm.shape) == 2:
        floatIm = newIm.astype(np.float64)
        floatIm /= PIXEL_MAX_INTENSITY
        return floatIm
    else:
        return rgb2gray(newIm)


def create_gaussian_filter(kernel_size):
    """

    :param kernel_size:
    :return:
    """
    base_gaussian_kernel = gaussian_kernel_1d = np.array([1, 1])
    if kernel_size == 1:
        gaussian_kernel_1d = np.array([1])
    else:
        for i in range(kernel_size - 2):
            gaussian_kernel_1d = signal.convolve(base_gaussian_kernel, gaussian_kernel_1d)

    returned_kernel = np.dot(gaussian_kernel_1d, 1 / np.sum(gaussian_kernel_1d[:, None]))
    return returned_kernel


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Task 3.1 a
    should return pyr as std array (NOT NUMPY!) with max len of max_levels, and each element of the array is a
    grayscale image.
    the filter_vec is an output which is 1D-row of size filter_size used for the pyramid construction.

    :param im:          a grayscale image with double values in [0, 1] (e.g. the output of ex1’s read_image with the
                        representation set to 1)
    :param max_levels:  the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
                        in constructing the pyramid filter (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25]).
    :return:            tuple of two values - pyr, filter_vec
    """
    pyr = []
    new_image = im
    pyr.append(new_image)
    filter_vec = create_gaussian_filter(filter_size)
    filter_vec = filter_vec.reshape(1, filter_size)
    filter_vec_t = filter_vec.transpose()

    for level in range(max_levels - 1):
        im_shape = new_image.shape
        if im_shape[0] <= 16 or im_shape[1] <= 16:
            break
        else:
            reduced_image = reduce(new_image, filter_vec, filter_vec_t)
            pyr.append(reduced_image)
            new_image = reduced_image

    return pyr, filter_vec

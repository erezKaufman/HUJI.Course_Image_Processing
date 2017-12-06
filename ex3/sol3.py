from skimage.transform import pyramid_gaussian, pyramid_laplacian
import os
import numpy as np

from scipy.ndimage.filters import convolve as convolve
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy.misc import imread
from scipy import signal

PIXEL_MAX_INTENSITY = 255


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def display_image_in_actual_size(im_data):
    dpi = 80
    height, width = im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    # plt.show()


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

    # gaussian_kernel_2d = np.zeros(kernel_size * kernel_size)
    # #
    # gaussian_kernel_2d = gaussian_kernel_2d.reshape(kernel_size, kernel_size)
    # gaussian_kernel_2d[int(kernel_size / 2)] = gaussian_kernel_1d
    # gaussian_kernel_2d_T = gaussian_kernel_2d.transpose()
    # # print(gaussian_kernel_2d_T)
    # gaussian_kernel = ndimage.filters.convolve(gaussian_kernel_2d, gaussian_kernel_2d_T)
    # # if is_expand:
    # #     gaussian_kernel = np.dot(gaussian_kernel, 1 / np.sum(gaussian_kernel[:, None]))
    #
    # gaussian_kernel = np.dot(gaussian_kernel, 1 / np.sum(gaussian_kernel[:, None]))
    returned_kernel = np.dot(gaussian_kernel_1d, 1 / np.sum(gaussian_kernel_1d[:, None]))
    return returned_kernel


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


def reduce(im, gaus_filter, gaus_filter_t):
    im_blured = convolve(im, gaus_filter)
    im_blured = convolve(im_blured, gaus_filter_t)
    return im_blured[::2, ::2]


def expand(im, filter_vec, filter_vec_t):
    new_size = np.array(im.shape) * 2
    zero_matrix = np.zeros(new_size, dtype=np.float64)
    zero_matrix[::2, ::2] = im
    im_blured = convolve(zero_matrix, filter_vec)
    im_blured = convolve(im_blured, filter_vec_t)
    return im_blured


# DONE
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
    # TODO check to see what I can do to check dimensions are multiples of 2**(max_levels−1)

    return pyr, filter_vec


# SEMI DONE
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

    # TODO check if the returned image is too bright
    return pyr, filter_vec


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


def render_pyramid(pyr, levels):
    """
    Task 3.3 a
    :param pyr:
    :param levels:
    :return:
    """
    if levels > len(pyr):
        levels = len(pyr)

    col_list = [pyr[0].shape[1]]

    # TODO check if for level = 3 need to return 4 images. AND CHECK IF levels-1 is good
    for i in range(levels - 1):
        col_list.append(pyr[i].shape[1] // 2)
    col = np.sum(col_list, 0)
    init_black_image = np.zeros((pyr[0].shape[0], col))
    init_index = 0

    for i in range(levels):
        current_pic = (pyr[i] - pyr[i].min()) / (pyr[i].max() - pyr[i].min())
        row, col = current_pic.shape
        # TODO - how to stretch image beteen 0 to 1?
        init_black_image[0:col, init_index:init_index + row] = current_pic
        # init_black_image[init_index:row, 0:col] = pyr[index]
        init_index += col_list[i]
    return init_black_image


def display_pyramid(pyr, levels):
    """
    Task 3.3 b
    :param pyr:
    :param levels:
    :return: None
    """
    plt.figure()  # TODO REMOVE THIS!

    image = render_pyramid(pyr, levels)
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.show()  # TODO RETURN THIS!


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
    # L1, vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    # L2, vec = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    L1= list(pyramid_laplacian(im1, max_layer=2, downscale=2))
    L2= list(pyramid_laplacian(im2, max_layer=2, downscale=2))
    Gm, vec = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)
    l_out = []
    for k in range(max_levels):
        first_product = (Gm[k]) * (L1[k])
        second_product = (1 - Gm[k]) * L2[k]
        result = first_product + second_product

        plt.imshow(result, cmap=plt.get_cmap('gray'))
        plt.show() # TODO RETURN THIS!
        l_out.append(result)
    return (laplacian_to_image(l_out, filter_vec, [1 for i in range(max_levels)]))
                     # TODO check the function


def blending_example1():
    """
    Task 4.1 a
    :return: im1, im2, mask, im_blend
    """
    image1 = read_image(relpath('jerusalem_sky1.jpg'), 1)
    image2 = read_image(relpath('aurora1.jpg'), 1)
    mask1 = read_image(relpath('mask1.jpg'), 1)
    mask1 = mask1 > 127
    # print(mask1)
    ret_image = pyramid_blending(image1, image2, mask1, 3, 3, 3)
    plt.imshow(ret_image, cmap=plt.get_cmap('gray'))
    plt.show()  # TODO RETURN THIS!


def blending_example2():
    """
    Task 4.1 b
    :return: im1, im2, mask, im_blend
    """
    pass


if __name__ == '__main__':
    image_path = "F:\My Documents\Google Drive\תואר ראשון מדמח\שנה ג\עיבוד תמונה\exs\HUJI.Course_Image_Processing\ex3\gray_orig.png"
    im = read_image(image_path, 1)
    pyr, filter_vec = build_laplacian_pyramid(im, 4, 3)
    image = im

    lap_pyr = list(pyramid_laplacian(image, max_layer=3, downscale=2))

    # for pic in pyr:
    #     display_image_in_actual_size(pic)

    # for pic, bpic in zip(pyr,lap_pyr):
    #     plt.figure()
    #     display_image_in_actual_size(pic)
    #     display_image_in_actual_size(bpic)
    #     plt.show()
    # display_pyramid(lap_pyr, 4)
    # display_pyramid(pyr, 4)
    # plt.show()
    blending_example1()
    # image = laplacian_to_image(pyr,filter_vec,[1,1,1])

    # for buikt_pyt, pyt in zip(lap_pyr,pyr):
    #     display_image_in_actual_size(buikt_pyt)
    #     display_image_in_actual_size(pyt)
    #
    # rows, cols = image.shape
    # pyramid = tuple(pyramid_gaussian(image, max_layer=2, downscale=2))
    #
    # for pyt in lap_pyr:
    #     display_image_in_actual_size(pyt)

    # composite_image = np.zeros((rows, cols + cols // 2), dtype=np.double)
    #
    # composite_image[:rows, :cols] = pyramid[0]

    # i_row = 0
    # for p in pyramid[1:]:
    #     n_rows, n_cols = p.shape[:2]
    #     composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
    #     i_row += n_rows
    #
    # fig, ax = plt.subplots()
    # ax.imshow(x`composite_image)

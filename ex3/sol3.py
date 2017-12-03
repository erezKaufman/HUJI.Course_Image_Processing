import numpy as np

import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy.misc import imread, comb
from scipy import signal, ndimage
PIXEL_MAX_INTENSITY = 255

def display_image_in_actual_size(im_data):

    dpi = 80
    height, width= im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()

def create_gaussian_kernel(kernel_size):
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

    gaussian_kernel_2d = np.zeros(kernel_size * kernel_size)
    #
    gaussian_kernel_2d = gaussian_kernel_2d.reshape(kernel_size, kernel_size)
    gaussian_kernel_2d[int(kernel_size / 2)] = gaussian_kernel_1d
    gaussian_kernel_2d_T = gaussian_kernel_2d.transpose()
    # print(gaussian_kernel_2d_T)
    gaussian_kernel = ndimage.filters.convolve(gaussian_kernel_2d, gaussian_kernel_2d_T)
    gaussian_kernel = np.dot(gaussian_kernel, 1 / np.sum(gaussian_kernel[:, None]))
    return gaussian_kernel_1d, gaussian_kernel



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


def reduce(im,gaussian_kernel):
    im_blured = ndimage.filters.convolve(im,gaussian_kernel)
    return im_blured[::2,::2]

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
    pyr.append(im)
    filter_vec , gaussian_kernel = create_gaussian_kernel(filter_size)
    for level in range(max_levels):
        im_shape = im.shape
        if im_shape[0] <= 16 or im_shape[1] <= 16:
            break
        else:
            reduced_image = reduce(im, gaussian_kernel)
            pyr.append(reduced_image)
            im = reduced_image
    # TODO check to see what I can do to check dimensions are multiples of 2**(max_levels−1)
    # for pic in pyr:
    #     display_image_in_actual_size(pic)

    return pyr, filter_vec


def  build_laplacian_pyramid(im, max_levels, filter_size):
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
    pass



def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    Task 3.2
    both lpyr and filter_vec are garuenteed from task 3.1
    :param lpyr:       Laplacian pyramid
    :param filter_vec: filter
    :param coeff:
    :return: img
    """
    pass


def render_pyramid(pyr, levels):
    """
    Task 3.3 a
    :param pyr:
    :param levels:
    :return:
    """
    pass


def display_pyramid(pyr, levels):
    """
    Task 3.3 b
    :param pyr:
    :param levels:
    :return: None
    """
    pass


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
    pass



def blending_example1():
    """
    Task 4.1 a
    :return: im1, im2, mask, im_blend
    """
    pass

def blending_example2():
    """
    Task 4.1 b
    :return: im1, im2, mask, im_blend
    """
    pass


if __name__ == '__main__':
    image_path = "/cs/usr/erez/Documents/image processing/exs/HUJI.Course_Image_Processing/ex3/gray_orig.png"
    im = read_image(image_path,1)
    pyr, filter_vec = build_gaussian_pyramid(im,3,3)
    # build_gaussian_pyramid(im, 3, 3)
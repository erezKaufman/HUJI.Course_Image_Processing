##################################################
#  FILE: sol1.py
#  WRITER : Erez Kaufman, erez, 305605255.
#  EXERCISE : Image Process ex1 2017-2018
#
##################################################
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

from scipy.misc import imread

PIXEL_MAX_INTENSITY = 255
MAGIC_RGB_YIQ_TRANSFORM_MATRIX = [[0.299, 0.587, 0.114],
                                  [0.596, -0.275, -0.321],
                                  [0.212, -0.523, 0.311]]

transMat = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
invTransMat = np.linalg.inv(transMat)


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


def imdisplay(filename, representation):
    """
    the functino will represent the image that we want to read from read_image
    :param filename:  a string that tells the image location
    :param representation:  an integer representing in what way we wish the image to appear.
    1 - gray scale
    2 - rgb
    :return: None
    """
    if representation == 2:
        plt.imshow(read_image(filename, representation))
    else:
        plt.imshow(read_image(filename, representation), cmap=plt.get_cmap('gray'))
    plt.show()


def rgb2yiq(imRGB):
    """
    A function that transforms the image from RGB to YIQ
    :param imRGB: a numpy image format
    :return: an image in YIQ presentation
    """
    return_img = imRGB.copy()[:, :, 0:3]
    conversion_matrix = np.array(MAGIC_RGB_YIQ_TRANSFORM_MATRIX)
    return_mat = np.dot(return_img, conversion_matrix.T)

    return return_mat


def yiq2rgb(imYIQ):
    """
    A function that transforms the image from YIQ to RGB
    :param imYIQ: a numpy image format
    :return: an image in RGB presentation
    """
    return_img = imYIQ.copy()[:, :, 0:3]
    conversion_matrix = np.array(MAGIC_RGB_YIQ_TRANSFORM_MATRIX)
    conversion_matrix = np.linalg.inv(conversion_matrix)

    returnMat = np.dot(return_img, conversion_matrix.T)
    return returnMat


def histogram_equalize_rgb(rgb_image):
    """
    the function changes the image into YIQ, calls the 'histogram_equalize_gray' function, and then changes the same
    image back to RGB. and returns the list as specified in 'histogram_equalize' documentation
    :param rgb_image:  a numpy image format
    :return: the list as specified in 'histogram_equalize' documentation
    """
    yiq_image = rgb2yiq(rgb_image)

    equlized_y_channel, original_hist, equlaized_hist = histogram_equalize_gray(yiq_image[:, :, 0])
    yiq_image[:, :, 0] = (equlized_y_channel )
    return np.clip(yiq2rgb(yiq_image),0,1), original_hist, equlaized_hist


def histogram_equalize_gray(gray_image):
    """
    the main work of the equalization is done in this function. first we calculate the histogram of the original image,
    then we calculate the cumulative histogram of the image, normalize it - and stretch its values to be from 0 to 256
    :param gray_image: a numpy image format
    :return: the list as specified in 'histogram_equalize' documentation
    """
    gray_level = (gray_image * 25).astype(np.uint8)

    # original image histogram to return
    hist_orig, yAxe = np.histogram(gray_level.flatten(), 256)

    cdf = np.cumsum(hist_orig)  # calculate the cdf of the original histogram
    cdf = cdf * yAxe[-2] / cdf[-1]  # normalize and multiply with the maximal gray level

    hist_image_flatten = np.interp(gray_level, yAxe[:-1], cdf)  # linearly stretch the values of the gray level
    im_eq = hist_image_flatten.reshape(gray_level.shape)  # reshape the new equalized image
    hist_eq, yAxe = np.histogram(hist_image_flatten.astype(np.uint8),
                                 256)  # calculate the histogram of the equalized image

    im_eq /= 255

    return np.clip(im_eq, 0, 1).astype(np.float64), hist_orig, hist_eq


def histogram_equalize(im_orig):
    """
    the function gets an numpy's image format and by the shape of it we can tell if it's an RGB or a grayscale image.
    with that knowledge we transfer the image to the right function to do an histogram equalization of its levels of
    colors.
    :return: a list -
    [0] - the equalized image.
    [1] - the original image's  histogram
    [2] = the equalized image's histogram
    """
    if len(im_orig.shape) == 3:
        return histogram_equalize_rgb(im_orig)
    else:
        return histogram_equalize_gray(im_orig)


def calculate_error(z_list, q_list, n_quan, image_histogram):
    """
    calculating the error in the specific iteration
    :param z_list: list of z's
    :param q_list: list of q's
    :param n_quan: number of quants in the run
    :param image_histogram: the histogram of the original image
    :return: number that represents the minimal error in the iteration
    """
    error_count = 0
    for i in range(n_quan):
        zi_segment = np.arange(z_list[i], z_list[i + 1] + 1)
        error_count += np.sum(np.square(q_list[i] - zi_segment) * image_histogram[z_list[i]:z_list[i + 1] + 1])
    return error_count


def create_z(image_histogram, n_quant):
    """
    creating initial z values
    :param image_histogram: the histogram of the original image
    :param n_quan: number of quants in the run
    :return: list of segments for 256 levels
    """
    cdf = image_histogram.cumsum()
    segments = np.arange(256) * (cdf[-1] / n_quant)
    z_list = (np.argmin(np.abs(cdf[:, np.newaxis] - segments), axis=0))[:n_quant + 1]
    z_list[0] = 0
    z_list[n_quant] = 255

    return z_list


def calculate_z(q_list, n_quan):
    """
    updating the z list using the current q_list
    :param q_list: list of q's
    :param n_quan: number of quants in the run
    :return: returning a new list of segments for 256 levels
    """
    new_z_list = np.array([0] * (n_quan + 1))

    for i in range(1, (n_quan)):
        new_z_list[i] = (q_list[i - 1] + q_list[i]) / 2
    new_z_list[-1] = 255
    return new_z_list


def calculate_q(z_list, n_quant, image_histogram):
    """
    calculate the desired level that we wish all the pixels in the specific segment will be
    :param z_list: list of segments
    :param n_quant: number of quants in the run
    :param image_histogram: the image histogram
    :return: return the list of the new q's
    """
    im_q = np.array([0] * n_quant)
    for index in range(len(z_list) - 1):
        zi_array = image_histogram[z_list[index]:z_list[index + 1] + 1]
        im_q[index] = np.sum(zi_array * np.arange(z_list[index], z_list[index + 1] + 1)) / np.sum(zi_array)

    return im_q.astype(np.int)


def quantize_gray(original_image, n_quant, n_iter):
    """
    the main function that do all the quantization process. we choose ourself segments, then choose the q's that will
    be the levels on each segment that the pixel will turn its colors to.
    :param original_image:
    :param n_quant:
    :param n_iter:
    :return:
    """
    error_list = []
    image_inlarge_255 = (original_image * 255).astype(np.uint8)
    image_histogram, yAxe = np.histogram(image_inlarge_255, 256)
    z_list = create_z(image_histogram, n_quant)
    q_list = calculate_q(z_list, n_quant, image_histogram)
    old_z = np.array([])
    for i in range(n_iter):
        if np.array_equal(z_list, old_z):
            break
        old_z = z_list.copy()
        z_list = calculate_z(q_list, n_quant)
        q_list = calculate_q(z_list, n_quant, image_histogram)
        error_list.append(calculate_error(z_list, q_list, n_quant, image_histogram))

    lut = np.array([0] * 256)
    for segment in range(len(q_list)):
        lut[z_list[segment]:z_list[segment + 1] + 1] = q_list[segment]
    quantize_image = lut[image_inlarge_255]

    return (quantize_image / 255).astype(np.float64), error_list


def quantize_rgb(im_orig, n_quant, n_iter):
    """

    :param im_orig:
    :param n_quant:
    :param n_iter:
    :return:
    """
    yiq_image = rgb2yiq(im_orig.copy())
    gray_quantization, error_list = quantize_gray(yiq_image[:, :, 0], n_quant, n_iter)
    yiq_image[:, :, 0] = gray_quantization

    return_image = yiq2rgb(yiq_image)
    return return_image, error_list


def quantize(im_orig, n_quant, n_iter):
    """
    :param im_orig:
    :param n_quant:
    :param n_iter:
    :return:
    """
    if len(im_orig.shape) == 3:
        return quantize_rgb(im_orig, n_quant, n_iter)
    else:
        return quantize_gray(im_orig, n_quant, n_iter)

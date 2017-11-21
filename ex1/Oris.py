# <editor-fold desc="Imports">

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread
from skimage.color import rgb2gray as rgb2gray

# </editor-fold>

# <editor-fold desc="Constants">

GRAYSCALE = 1
RGB = 2

RGB2YIQ_MAT = np.array([
    [0.299, 0.587, 0.114],
    [0.596, -0.275, -0.321],
    [0.212, -0.523, 0.311]
])

YIQ2RGB_MAT = np.linalg.inv(RGB2YIQ_MAT)


# </editor-fold>

# Done
# <editor-fold desc="3.1 : Read Image">
def read_image(filename: str, representation: int) -> np.ndarray:
    """
    read a file in a given path and given representation.
    :param filename: path.
    :type filename: str
    :param representation: Grayscale (1), or RGB (2)
    :type representation: int
    :return: the image which was read.
    """
    if representation == GRAYSCALE:
        return rgb2gray(imread(filename))
    elif representation == RGB:
        return (imread(filename) / 255).astype(np.float64)


# </editor-fold>

# Done
# <editor-fold desc="3.2 : Show Image">
def imdisplayImage(im: np.ndarray, representation: int) -> None:
    """
    get image and display it in given representation to user.
    :param im: the image matrix.
    :type im: np.ndarray
    :param representation: Grayscale (1), or RGB (2)
    :type representation: int
    :return: None
    """
    plt.figure()
    if (representation == GRAYSCALE):
        plt.imshow(np.clip(im, 0, 1), cmap=plt.cm.gray)
    else:
        plt.imshow(np.clip(im, 0, 1))
    plt.show()


def imdisplay(filename: str, representation: int) -> None:
    """
    get image path and display it in given representation to user.
    :param filename: the filename path.
    :type im: str
    :param representation: Grayscale (1), or RGB (2)
    :type representation: int
    :return: None
    """
    imdisplayImage(read_image(filename, representation), representation)


# </editor-fold>

# Done
# <editor-fold desc="3.3 : RGB <-> YIQ">
def convert(im: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    convert im to other repr using matrix multiplication.
    :param im: the image to convert
    :type im: np.ndarray
    :param matrix: the matrix to convert with.
    :type matrix: np.ndarray
    :return: the converted image
    """
    return np.dot(im, matrix.T)


def rgb2yiq(imRGB: np.ndarray) -> np.ndarray:
    """
    convert im to YIQ repr.
    :param imRGB: the image to convert
    :type imRGB: np.ndarray
    :return: the converted image
    """
    return convert(imRGB, RGB2YIQ_MAT)


def yiq2rgb(imYIQ: np.ndarray) -> np.ndarray:
    """
    convert im to RGB repr.
    :param imYIQ: the image to convert
    :type imYIQ: np.ndarray
    :return: the converted image
    """
    return convert(imYIQ, YIQ2RGB_MAT)


# </editor-fold>

# Done
# <editor-fold desc="3.4 : Histogram Equalize">
def histogram_equalize(im_orig: np.ndarray):
    """
    perform histogram equalize on given image.
    :param im_orig: the image to be equalized.
    :type im_orig: np.ndarray
    :return: the equalized image, former hist, current hist.
    """
    if im_orig.ndim == 2:
        return histogram_equalize_grayscale(im_orig)
    else:
        return histogram_equalize_rgb(im_orig)


def histogram_equalize_rgb(im: np.ndarray):
    """
    perform histogram equalize on RGB format image.
    :param im: the image to be equalized.
    :type im: np.ndarray
    :return: the equalized image, former hist, current hist.
    """
    yiq_im = rgb2yiq(im)
    y_eq, image_histogram, new_image_histogram = histogram_equalize_grayscale(yiq_im[:, :, 0])
    yiq_im[:, :, 0] = y_eq
    plt.plot(new_image_histogram)
    # plt.xlim([0, 256])

    plt.show()
    return yiq2rgb(yiq_im), image_histogram, new_image_histogram


def histogram_equalize_grayscale(im: np.ndarray):
    """
    perform histogram equalize on Grayscale format image.
    :param im: the image to be equalized.
    :type im: np.ndarray
    :return: the equalized image, former hist, current hist.
    """
    im256 = (im * 255).astype(np.uint8)
    image_histogram, bins = np.histogram(im256, 256)
    cumulative_histogram = np.cumsum(image_histogram)
    cumulative_histogram_normalized = 255 * (cumulative_histogram / cumulative_histogram[-1])
    image_equalized = np.interp(im256, bins[:-1], cumulative_histogram_normalized)
    image_equalized = image_equalized.reshape(im.shape)
    new_image_histogram, bins = np.histogram(image_equalized.astype(np.uint8), 256)
    cdf_eq_hist = np.cumsum(new_image_histogram)


    return np.clip((image_equalized).astype(np.float64) / 255, 0, 1), image_histogram, new_image_histogram


# </editor-fold>

# Done
# <editor-fold desc="3.5 : Quantize">
def quantize(im_orig: np.ndarray, n_quant: int, n_iter: int) -> (np.ndarray, list):
    """
    perform optimized quantize on given image with given quants number and iter number.
    :param im_orig: the image to be quantize.
    :type im_orig: np.ndarray
    :param n_quant: the amount of quants.
    :type n_quant: int
    :param n_iter: the max iter count.
    :type n_iter: int
    :return: the quantize image.
    """
    if im_orig.ndim == 2:
        return quantize_grayscale(im_orig, n_quant, n_iter)
    else:
        return quantize_rgb(im_orig, n_quant, n_iter)


def quantize_grayscale(im_orig: np.ndarray, n_quant: int, n_iter: int) -> (np.ndarray, list):
    """
    perform optimized quantize on given grayscale image with given quants number and iter number.
    :param im_orig: the image to be quantize.
    :type im_orig: np.ndarray
    :param n_quant: the amount of quants.
    :type n_quant: int
    :param n_iter: the max iter count.
    :type n_iter: int
    :return: the quantize image , error.
    """
    im256 = (im_orig * 255).round().astype(np.uint8)
    image_histogram, bins = np.histogram(im256.flatten(), 256)
    z = init_z(image_histogram, n_quant)
    qs = np.array([0] * (z.size - 1), dtype=int)
    # print(z)
    update_q(qs, z, image_histogram)
    # print(qs    )
    formerZ = np.array([])
    error = []
    for iter_counter in range(n_iter):
        print(z)
        if np.array_equal(formerZ, z):
            break
        formerZ = z.copy()
        update_z(z, qs)
        update_q(qs, z, image_histogram)
        error.append(calc_error(z, qs, image_histogram))

    quantize_map = np.array([0] * 256)
    for i in range(n_quant):
        quantize_map[z[i]:z[i + 1]] = qs[i]
    quantize_image = quantize_map[im256]

    return quantize_image.astype(np.float64) / 255, error


def init_z(image_histogram: np.ndarray, n_quant: int) -> np.ndarray:
    """
    init not empty z division
    :param image_histogram: the image histogram to deduce from.
    :param n_quant: number of quants.
    :return: division of the segment 0-255
    """
    z = np.array([0] * (n_quant + 1), dtype=int)
    cumulative_histogram = np.cumsum(image_histogram)
    avg_pxl_value = cumulative_histogram[-1] / n_quant
    for i in range(1, n_quant):
        z[i] = np.where(cumulative_histogram >= i * avg_pxl_value)[0][0]
    z[n_quant] = 255
    return z


def update_z(z: np.ndarray, qs: np.ndarray) -> None:
    """
    update Z per iter, update done by the QS.
    :param z: the z to update.
    :param qs: the q to update by.
    :return: None.
    """
    qs_cycle = np.roll(qs, -1)
    z[1:-1] = np.round((qs[:-1] + qs_cycle[:-1]) / 2)


def update_q(qs: np.ndarray, z: np.ndarray, image_histogram: np.ndarray) -> None:
    """
    update the QS using Z and image histogram.
    :param qs: QS to update.
    :param z:  Z to update by.
    :param image_histogram: histogram to update by.
    :return: None.
    """
    for i in range(qs.size):
        segment = np.arange(z[i], z[i + 1] + 1)
        # print(segment)
        pz = image_histogram[segment]
        # print(pz)
        qs[i] = np.sum(segment * pz) / np.sum(pz)

    # print(qs)


def calc_error(z: np.ndarray, qs: np.ndarray, image_histogram: np.ndarray) -> int:
    """
    calc the square loss of the equalize process.
    :param z: the Z to get the segments from.
    :param qs:  the QS
    :param image_histogram:  the image histogram.
    :return: error value.
    """
    total = 0
    for i in range(qs.size):
        segment = np.arange(z[i], z[i + 1] + 1)
        total += (np.square(segment - qs[i]) * image_histogram[segment]).sum()
    return total


# </editor-fold>

# Done
# <editor-fold desc="Bonus">

def quantize_rgb(im_orig: np.ndarray, n_quant: int, n_iter: int) -> (np.ndarray, list):
    """
    perform K means algorithm on given grayscale image with given quants number and iter number.
    :param im_orig: the image to be quantize.
    :type im_orig: np.ndarray
    :param n_quant: the amount of quants.
    :type n_quant: int
    :param n_iter: the max iter count.
    :type n_iter: int
    :return: the quantize image , error.
    """
    from scipy.cluster import vq  # local import so won't affect other solutions.
    reshaped_im = im_orig.reshape(im_orig.shape[0] * im_orig.shape[1], 3)
    error = []
    centroids, label = vq.kmeans2(reshaped_im, k=n_quant, iter=1)
    former_centroids = np.ndarray(centroids.shape)
    error.append(calc_error_rgb(im_orig, centroids, label))
    for i in range(n_iter - 1):
        if np.array_equal(centroids, former_centroids):
            break
        former_centroids = centroids.copy()
        centroids, label = vq.kmeans2(data=reshaped_im, k=former_centroids, iter=1, minit="matrix")
        error.append(calc_error_rgb(im_orig, centroids, label))
    reshaped_im = centroids[label]
    im_quant = reshaped_im.reshape(im_orig.shape[0], im_orig.shape[1], 3)
    return im_quant, error


def calc_error_rgb(im_orig: np.ndarray, centroids: np.ndarray, label: np.ndarray) -> np.float64:
    """
    calc L2 loss on RGB image.
    :param im_orig: the src image.
    :param centroids: the centroids of the K means
    :param label:  the label of the K means.
    :return: the error value.
    """
    reshaped_im = centroids[label]
    im_quant = reshaped_im.reshape(im_orig.shape[0], im_orig.shape[1], 3)
    diff = np.sqrt(np.sum(np.square(im_quant - im_orig), axis=2))  # sqrt of sum of squared diff rgb diff.
    total = np.sum(np.sum(diff, axis=0))  # sum each axis due to 2D matrix
    return total.astype(np.float64)

# </editor-fold>

if __name__ == '__main__':
    quantize(read_image('F:\My Documents\Google Drive\תואר ראשון מדמח\שנה ג\עיבוד '
                        'תמונה\exs\ex1\quantization_examples\\rgb_orig.png',2),10,100)
    # quantize(read_image('F:\My Documents\Google Drive\תואר ראשון מדמח\שנה ג\עיבוד תמונה\exs\ex1\quantization_examples\gray_orig.png',2),10,16)
    # quantize(read_image('F:\My Documents\Google Drive\תואר ראשון מדמח\שנה ג\עיבוד '
    #                     'תמונה\exs\ex1\quantization_examples\\rgb_orig.png', 2), 10, 100)
    # plt.show()
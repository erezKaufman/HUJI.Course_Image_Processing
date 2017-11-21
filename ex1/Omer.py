from scipy.misc import imread as imread
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np
from skimage.exposure import equalize_hist
import copy

GRAY_SCALE = 1

transMat = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
invTransMat = np.linalg.inv(transMat)


def read_image(filename, representation):
    if representation != GRAY_SCALE:
        return imread(filename, False).astype(np.float64) / 255
    else:
        return rgb2gray(imread(filename, True)).astype(np.float64) / 255


def imdisplay(filename, representation):
    im = read_image(filename=filename, representation=representation)

    if representation == GRAY_SCALE:
        plt.imshow(im, cmap='gray')
    else:
        plt.imshow(im)

    plt.show()


def rgb2yiq(imRGB):
    r, g, b = imRGB[:, :, 0], imRGB[:, :, 1], imRGB[:, :, 2]
    resIm = np.zeros(imRGB.shape)

    resIm[:, :, 0] = transMat[0, 0] * r + transMat[0, 1] * g + transMat[0, 2] * b
    resIm[:, :, 1] = transMat[1, 0] * r + transMat[1, 1] * g + transMat[1, 2] * b
    resIm[:, :, 2] = transMat[2, 0] * r + transMat[2, 1] * g + transMat[2, 2] * b
    return resIm


def yiq2rgb(imYIQ):
    r, g, b = imYIQ[:, :, 0], imYIQ[:, :, 1], imYIQ[:, :, 2]
    resIm = np.zeros(imYIQ.shape)

    resIm[:, :, 0] = invTransMat[0, 0] * r + invTransMat[0, 1] * g + invTransMat[0, 2] * b
    resIm[:, :, 1] = invTransMat[1, 0] * r + invTransMat[1, 1] * g + invTransMat[1, 2] * b
    resIm[:, :, 2] = invTransMat[2, 0] * r + invTransMat[2, 1] * g + invTransMat[2, 2] * b

    return resIm


def isRGBImage(im):
    return len(im.shape) == 3


def doEqualizationGrayScale(gray_image):
    gray_level = (gray_image * 255).astype(np.uint8)

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
    if isRGBImage(im_orig):
        yiq_image = rgb2yiq(im_orig)
        equlized_y_channel, original_hist, equlaized_hist = doEqualizationGrayScale(yiq_image[:, :, 0])
        yiq_image[:, :, 0] = (equlized_y_channel / 255)
        # plt.imshow(yiq_image,cmap=plt.get_cmap('gray'))
        plt.imshow(yiq2rgb(yiq_image))
        plt.show()
        # return np.clip(yiq2rgb(im_yiq), 0, 1), old_hist, new_hist
    else:
        return doEqualizationGrayScale(im_orig)


def findZs(original_hist, n_quant):
    im_z = np.array([0] * (n_quant + 1), dtype=int)
    cumulative_histogram = np.cumsum(original_hist)
    avg_pxl_value = cumulative_histogram[-1] / n_quant
    for i in range(1, n_quant):
        im_z[i] = np.where(cumulative_histogram >= i * avg_pxl_value)[0][0]
    im_z[n_quant] = 255
    im_z[0] = 0.0
    return im_z


def updateZs(im_q, n_quant):
    im_z = np.array([0] * (n_quant + 1))

    im_z[0] = 0
    im_z[n_quant] = 255
    for zi in range(1, n_quant):
        im_z[zi] = (im_q[zi - 1] + im_q[zi]) / 2

    return im_z


def findQs(original_hist, n_quant, im_z):
    im_q = np.ndarray(n_quant)
    for zi in range(n_quant):
        zi_segment = np.array(original_hist[im_z[zi]:im_z[zi + 1] + 1])
        im_q[zi] = np.sum(zi_segment * range(im_z[zi], im_z[zi + 1] + 1, 1)) / np.sum(zi_segment)
    return im_q.astype(np.int)


def findError(original_hist, n_quant, im_q, im_z):
    error = 0
    for zi in range(n_quant):
        zi_segment = np.arange(im_z[zi], im_z[zi + 1] + 1)
        error += np.sum(np.square(im_q[zi] - zi_segment) * original_hist[im_z[zi]:im_z[zi + 1] + 1])
    return error


def grayQuantize(im_gray, n_quant, n_iter):
    im_int = (im_gray * 255).astype(np.uint8)
    original_hist, im_bin = np.histogram(im_gray.flatten(), 256)
    error = np.array([0] * n_iter)
    im_z = findZs(original_hist, n_quant)
    im_q = findQs(original_hist, n_quant, im_z)
    for i in range(n_iter):
        old_im_z = im_z
        old_im_q = im_q
        error[i] = findError(original_hist, n_quant, im_q, im_z)
        im_z = updateZs(im_q, n_quant)
        im_q = findQs(original_hist, n_quant, im_z)
        if (np.array_equal(im_z, old_im_z)):
            break

    look_up_table = np.array([0] * 256)
    for zi in range(n_quant):
        look_up_table[im_z[zi]:im_z[zi + 1] + 1] = im_q[zi]

    return look_up_table[im_int].astype(np.float32) / 255


def quantize(im_orig, n_quant, n_iter):
    errors = ''
    if (isRGBImage(im_orig)):
        im_yiq = rgb2yiq(im_orig)
        temp = grayQuantize(im_yiq[:, :, 0], n_quant, n_iter)
        im_yiq[:, :, 0] = temp
        return yiq2rgb(im_yiq), errors

    else:
        return grayQuantize(im_orig, n_quant, n_iter), errors


# startingImRGB = read_image("./tests/quantization_examples/rgb_orig.png", 2)
#
# startingImGRAY = read_image("./tests/quantization_examples/gray_orig.png", 1)


# plt.figure(1)
# plt.imshow(quantize(startingImGRAY, 10, 100),cmap=plt.get_cmap('gray'))
# plt.figure(2)
# plt.imshow(quantize(startingImRGB, 10, 100))
# plt.figure(3)
# plt.imshow(histogram_equalize(startingImGRAY)[0],cmap=plt.get_cmap('gray'))
# plt.figure(4)
if __name__ == '__main__':
    rgb_3 = histogram_equalize(read_image('F:\My Documents\Google Drive\תואר ראשון מדמח\שנה ג\עיבוד '
                                          'תמונה\exs\ex1\quantization_examples\\rgb_orig.png', 2))[0]
    plt.imshow(rgb_3)
    plt.show()

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



def calculate_dft_matrix(column_to_N, array_to_N, negative, N):
    """
    calculating the dft matrix for the dft process
    :param column_to_N:
    :param array_to_N:
    :param negative:
    :param N:
    :return:
    """
    return np.exp((negative * 2j * np.pi * column_to_N * array_to_N) / N).astype(np.complex128)


def DFT(signal):
    """

    :param signal:
    :return:
    """
    N = signal.shape[0]
    array_to_N = np.arange(N)
    column_to_N = array_to_N.reshape(N, 1)
    dft_matrix = calculate_dft_matrix(column_to_N, array_to_N, (-1), N)
    return np.dot(dft_matrix, signal).astype(np.complex128)


def IDFT(fourier_signal):
    """

    :param fourier_signal:
    :return:
    """
    N = fourier_signal.shape[0]
    array_to_N = np.arange(N)
    column_to_N = array_to_N[:, np.newaxis]
    dft_matrix = calculate_dft_matrix(column_to_N, array_to_N, 1, N)
    return (np.dot(dft_matrix, fourier_signal)) / N


def DFT2(image):
    return DFT(DFT(image).transpose()).transpose()


def IDFT2(fourier_image):
    return IDFT(IDFT(fourier_image).transpose()).transpose()


def calc_magnitude(dx, dy):
    return np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)


def conv_der(im):
    x_matrix = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
    y_conv = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])

    x_con = signal.convolve2d(im, x_matrix, mode="same")
    y_conv = signal.convolve2d(im, y_conv, mode="same")

    magn = calc_magnitude(x_con, y_conv).astype(np.float64).astype(np.float64)

    return magn


def fourier_der(im):
    # 2.2
    # first - shift the image.
    N = im.shape[0]
    M = im.shape[1]
    u = np.arange(-N / 2, N / 2)
    v = np.arange(-M / 2, M / 2)
    u_shifted = np.fft.fftshift(u).reshape((N, 1))
    v_shifted = np.fft.fftshift(v).reshape((1,M))
    dft_image = DFT2(im)

    # second, calculate it's DFT
    # third - multiply with constants
    x_derived = IDFT2((2j * np.pi / N) * (u_shifted * dft_image))

    # plt.imshow(x_derived.astype(np.float64), cmap=plt.get_cmap('gray'))
    # plt.show()
    y_derived = IDFT2((2j * np.pi / M) * v_shifted * dft_image)
    # plt.imshow(y_derived.astype(np.float64),cmap=plt.get_cmap('gray'))
    # plt.show()
    # plt.imshow(magn.astype(np.float64), cmap=plt.get_cmap('gray'))
    # plt.show()
    magn = calc_magnitude(x_derived, y_derived).astype(np.float64)
    return magn.astype(np.float64)


def blur_spatial(im, kernel_size):
    """

    :param im:
    :param kernel_size:
    :return:
    """
    # 3.1






    gaussian_kernel = create_gaussian_kernel(kernel_size)
    return_val = signal.convolve2d(im, gaussian_kernel, mode="same")

    return return_val.astype(np.float64)


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
    gaussian_kernel = signal.convolve2d(gaussian_kernel_2d, gaussian_kernel_2d_T, 'same')
    gaussian_kernel = np.dot(gaussian_kernel, 1 / np.sum(gaussian_kernel[:, None]))
    return gaussian_kernel


def blur_fourier(im, kernel_size):
    """
    # 3.2

    :param im:
    :param kernel_size:
    :return:
    """

    N_to_pad = im.shape[0] - kernel_size
    M_to_pad = im.shape[1] - kernel_size
    gaussian_kernel = create_gaussian_kernel(kernel_size)

    if kernel_size % 2 == 1:
        gaus_test = np.lib.pad(gaussian_kernel,
                               ((np.floor(N_to_pad / 2), np.floor(N_to_pad / 2) + 1),
                                (np.floor(M_to_pad / 2), np.floor(M_to_pad / 2) + 1)),
                               mode='constant', constant_values=0)
    else:
        gaus_test = np.lib.pad(gaussian_kernel,
                               ((np.floor(N_to_pad / 2), np.floor(N_to_pad / 2)), (np.floor(M_to_pad / 2),
                                                                                   np.floor(M_to_pad /
                                                                                            2))),
                               mode='constant', constant_values=0)
    gaussian_shifted = np.fft.fftshift(gaus_test)
    dft_gaussian_kernel = DFT2(gaussian_shifted)
    dft_im = DFT2(im)
    inner_product = np.multiply(dft_gaussian_kernel, dft_im)
    inverse_dft_inner_product = IDFT2(inner_product)
    return inverse_dft_inner_product.astype(np.float64)


if __name__ == '__main__':
    # a = np.array([1,2,3,4,5,6])
    # a = a[:,np.newaxis]
    # image = (imread('F:\My Documents\Google Drive\תואר ראשון מדמח\שנה ג\עיבוד '
    #                 'תמונה\exs\HUJI.Course_Image_Processing\ex2\monkey.jpg'))
    image = read_image("external/monkey.jpg", 1)

    image = rgb2gray(image)
    # print(image.shape)
    # a = conv_der(image)
    # b = fourier_der(image)
    # plt.imshow(a,cmap=plt.get_cmap('gray'))
    # plt.show()
    # plt.imshow(b,cmap=plt.get_cmap('gray'))
    # plt.show()
    b = fourier_der(image)
    # print(np.allclose(a,b))
    a = blur_spatial(image, 9)
    # b = blur_fourier(image, 9)
    plt.figure()
    plt.imshow(a, cmap=plt.get_cmap('gray'))
    plt.figure()
    # c = np.fft.fftshift(c)
    plt.imshow(b, cmap=plt.get_cmap('gray'))
    print("real:\n", a)
    # print(np.array_equal(b, c))
    # a = np.arange(-3,3)
    plt.show()
    # a_s = np.f
    # ft.fftshift(a)
    # print(a_s)

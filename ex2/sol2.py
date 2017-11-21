import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy.misc import imread
from scipy import signal


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
    x_con = signal.convolve2d(im, np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]]))
    y_conv = signal.convolve2d(im, np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]]))
    # plt.imshow((fourier_x_conv) ,cmap=plt.get_cmap('gray'))
    # plt.show()
    # plt.imshow((fourier_y_conv), cmap=plt.get_cmap('gray'))
    # plt.show()
    # magnitude = np.sqrt(np.abs(x_con)**2 + np.abs(y_conv)**2)
    # plt.imshow((magnitude), cmap=plt.get_cmap('gray'))
    # plt.show()
    return calc_magnitude(x_con, y_conv).astype(np.float64)


def fourier_der(im):
    # 2.2
    # first - shift the image.
    N = im.shape[0]
    M = im.shape[1]
    u = np.arange(-N/2,N/2)
    v = np.arange(-M/2,M/2)
    u_shifted = np.fft.fftshift(u)
    v_shifted = np.fft.fftshift(v)
    dft_image = DFT2(im)

    # second, calculate it's DFT
    # third - multiply with constants
    x_derived = IDFT2((2j * np.pi / N) * u_shifted * dft_image)

    ##############
    # x_derived_s = IDFT2((2j * np.pi / N) * u_shifted * dft_image)
    # plt.imshow(x_derived_s.astype(np.float64), cmap=plt.get_cmap('gray'))
    # plt.show()
    ##############

    plt.imshow(x_derived.astype(np.float64), cmap=plt.get_cmap('gray'))
    plt.show()
    y_derived = IDFT2((2j * np.pi / M) * v_shifted * dft_image)
    plt.imshow(y_derived.astype(np.float64),cmap=plt.get_cmap('gray'))
    plt.show()
    magn = calc_magnitude(x_derived, y_derived)
    plt.imshow(magn.astype(np.float64), cmap=plt.get_cmap('gray'))
    plt.show()
    pass


def blur_spatial(im, kernel_size):
    # 3.1
    pass


def blur_fourier(im, kernel_size):
    # 3.2
    pass


if __name__ == '__main__':
    # a = np.array([1,2,3,4,5,6])
    # a = a[:,np.newaxis]
    image = (imread('/cs/usr/erez/Documents/image processing/exs/HUJI.Course_Image_Processing/ex2/gray_orig.png'))


    fourier_der(image)

    # a = np.arange(-3,3)
    # a_s = np.fft.fftshift(a)
    # print(a_s)
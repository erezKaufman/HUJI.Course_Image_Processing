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


    magn = calc_magnitude(x_con, y_conv).astype(np.float64)
    # plt.imshow((magn), cmap=plt.get_cmap('gray'))
    # plt.show()
    return magn


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

    # plt.imshow(x_derived.astype(np.float64), cmap=plt.get_cmap('gray'))
    # plt.show()
    y_derived = IDFT2((2j * np.pi / M) * v_shifted * dft_image)
    # plt.imshow(y_derived.astype(np.float64),cmap=plt.get_cmap('gray'))
    # plt.show()
    magn = calc_magnitude(x_derived, y_derived)
    # plt.imshow(magn.astype(np.float64), cmap=plt.get_cmap('gray'))
    # plt.show()
    return magn.astype(np.float64)


def blur_spatial(im, kernel_size):
    """

    :param im:
    :param kernel_size:
    :return:
    """
    # 3.1
    base_gaussian_kernel = gaussian_kernel_1d= np.array([1,1])
    for i in range(kernel_size-2):
        gaussian_kernel_1d = signal.convolve(base_gaussian_kernel,gaussian_kernel_1d)

    gaussian_kernel_2d = np.zeros(kernel_size*kernel_size)
    gaussian_kernel_2d = gaussian_kernel_2d.reshape(kernel_size,kernel_size)
    gaussian_kernel_2d[int(kernel_size/2)] =gaussian_kernel_1d
    gaussian_kernel_2d_T = gaussian_kernel_2d.transpose()

    # print(gaussian_kernel_2d_T)
    gaussian_kernel = signal.convolve2d(gaussian_kernel_2d,gaussian_kernel_2d_T,'same')

    gaussian_kernel = np.dot(gaussian_kernel,1/np.sum(gaussian_kernel[:,None]))
    return_val = signal.convolve2d(gaussian_kernel,im)

    plt.imshow(return_val, cmap=plt.get_cmap('gray'))
    plt.show()    # print(gaussian_kernel)

def blur_fourier(im, kernel_size):
    """
    # 3.2

    :param im:
    :param kernel_size:
    :return:
    """


if __name__ == '__main__':
    # a = np.array([1,2,3,4,5,6])
    # a = a[:,np.newaxis]
    image = (imread('/cs/usr/erez/Documents/image processing/exs/HUJI.Course_Image_Processing/ex2/gray_orig.png'))

    # a = conv_der(image)
    # b = fourier_der(image)
    # print(np.allclose(a,b)) # TODO check if the conv_der should return an image of shape (256,256) or (258,258
    blur_spatial(image,15)
    # a = np.arange(-3,3)
    # a_s = np.f
    # ft.fftshift(a)
    # print(a_s)
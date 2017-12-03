import numpy as np
from scipy import signal
import scipy.stats as st
from scipy import misc
from scipy.misc import imread as imread
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

kernel = [1, 0, -1]
GRAY_SCALE = 1
RGB = 2
NORMALIZE_IMAGE_VALUES = 255


def read_image(filename, representation=1):
    """
    reads an image from a file path (gray scale or RGB)
    :param filename:  the file name
    :param representation: the representation we choose 1 for Gray and 2 for RGB
    :return: a float64 image
    """
    if representation == GRAY_SCALE:
        return rgb2gray(imread(filename, True)).astype(np.float64)
    if representation == RGB:
        return imread(filename, False).astype(np.float64) / NORMALIZE_IMAGE_VALUES



def getTransformMatrix(N):
    """
    returns a transformation matrix size of N
    :param N: the size of the matrix
    :return: a fourier transformation matrix
    """
    x = np.arange(N)
    u = x.reshape((N, 1))
    transform_matrix = np.exp(-2j * np.pi * u * x / N)
    return transform_matrix


def DFT(signal):
    """
    transforms the signal to fourier_signal
    :param signal: the signal to transform
    :return: a fourier representation of the signal
    """
    N = signal.shape[0]
    transform_matrix = getTransformMatrix(N).astype(np.complex128)
    return np.dot(transform_matrix, signal).astype(np.complex128)


def getInverseTransformMatrix(N):
    """
    returns the inverse transformation matrix
    :param N: the size of the inverse transformation matrix
    :return: the inverse transformation matrix
    """
    x = np.arange(N)
    u = x.reshape((N, 1))
    transform_matrix = np.exp(2j * np.pi * u * x / N) / N
    return transform_matrix


def IDFT(fourier_signal):
    """
    transforms the fourier_signal to signal
    :param fourier_signal: the fourier signal to transform
    :return: the original signal of the fourier_signal
    """
    N = fourier_signal.shape[0]
    inv_trans_matrix = getInverseTransformMatrix(N).astype(np.complex128)
    return np.dot(inv_trans_matrix, fourier_signal).astype(np.complex128)


def DFT2(image):
    """
    transforms the image to fourier_image
    :param image: the image to transform
    :return: a fourier representation of the image
    """
    return DFT(DFT(image).transpose()).transpose()


def IDFT2(fourier_image):
    """
    transforms the fourier_image to the original image
    :param fourier_image: the fourier_image to transform
    :return: the original signal of the fourier_image
    """
    return IDFT(IDFT(fourier_image).transpose()).transpose()


def getMagnitude(dx, dy):
    """
    returns the magnitude of dx and dy derivative of an image
    :param dx: The x axis derivative of an image
    :param dy: The y axis derivative of an image
    :return: the magnitude of the gradient vector
    """
    return np.sqrt(np.abs(dx) * 2 + np.abs(dy) * 2)


def conv_der(im):
    """
    computes the derivative of the image and returns its magnitude
    :param im: the image to compute on
    :return: the derivatives magnitude
    """
    padded_kernel_row = np.asarray([[1, 0, -1]])
    padded_kernel_colum = padded_kernel_row.transpose()

    dx = signal.convolve2d(im, padded_kernel_row, mode='same')
    dy = signal.convolve2d(im, padded_kernel_colum, mode='same')

    return getMagnitude(dx, dy)


def fourier_der(im):
    """
    computes the fourier derivative of the image and returns its magnitude
    :param im: the image to compute on
    :return: the fourier derivatives magnitude
    """
    N = im.shape[0]
    F = DFT2(im)
    U = np.concatenate((np.arange(np.floor(N / 2) + 1), np.flip((-1 * np.arange(1, np.ceil(N / 2))), 0)))

    dx = IDFT2(F * U.reshape(N,1))

    M = im.shape[1]
    V = np.concatenate((np.arange(np.floor(M / 2) + 1), np.flip((-1 * np.arange(1, np.ceil(M / 2))), 0)))
    dy = IDFT2(F * V.reshape(1,M))

    return getMagnitude(dx, dy)


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
    if(kernel_size==1):
        return im
    guass_kernel = getGaussianKernel(kernel_size)
    return signal.convolve2d(im, guass_kernel,mode='same',boundary='fill',fillvalue=0)

def blur_fourier(im, kernel_size):
    """
    blurs an image with a guassian kernel
    :param im: the image to blur
    :param kernel_size: the kernel size
    :return: a blurred image
    """
    if(kernel_size==1):
        return im
    M, N = im.shape
    kernel = getGaussianKernel(kernel_size)
    X = M // 2 + 1
    Y = N // 2 + 1
    offset = kernel_size // 2
    G = np.zeros(im.shape)

    G[X - offset:X + offset + 1, Y - offset:Y + offset + 1] = kernel
    G = np.fft.ifftshift(G)
    fourier_im = DFT2(im)
    return np.real(IDFT2(np.multiply(fourier_im,G)))
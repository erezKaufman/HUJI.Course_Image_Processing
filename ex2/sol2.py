import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy.misc import imread

def calculate_dft_matrix(column_to_N,array_to_N,negative,N):
    """
    calculating the dft matrix for the dft process
    :param column_to_N:
    :param array_to_N:
    :param negative:
    :param N:
    :return:
    """
    return np.exp((negative*2j * np.pi * column_to_N * array_to_N) / N).astype(np.complex128)
def DFT(signal):
    """

    :param signal:
    :return:
    """
    N = signal.shape[0]
    array_to_N = np.arange(N)
    column_to_N = array_to_N.reshape(N,1)
    dft_matrix = calculate_dft_matrix(column_to_N,array_to_N,(-1),N)
    return np.dot(dft_matrix,signal).astype(np.complex128)

def IDFT(fourier_signal):
    """

    :param fourier_signal:
    :return:
    """
    N = fourier_signal.shape[0]
    array_to_N = np.arange(N)
    column_to_N = array_to_N[:, np.newaxis]
    dft_matrix = calculate_dft_matrix(column_to_N, array_to_N, 1, N)
    return (np.dot(dft_matrix,fourier_signal))/N



def DFT2(image):

    return DFT(DFT(image).transpose()).transpose()


def IDFT2(fourier_image):
    return IDFT(IDFT(fourier_image))


if __name__ == '__main__':
    # a = np.array([1,2,3,4,5,6])
    # a = a[:,np.newaxis]
    # image = (imread('F:\My Documents\Google Drive\תואר ראשון מדמח\שנה ג\עיבוד '
    #                 'תמונה\exs\HUJI.Course_Image_Processing\ex2\gray_orig.png'))
    a = np.array([a for a in range(0,8*511,8)])
    b = np.cumsum(a)
    plt.hist(b,256)
    plt.show()
    print(b)
    # plt.imshow(image ,cmap=plt.get_cmap('gray'))
    # plt.show()
    # DFT2(image )
    # # print (IDFT(DFT(a)))
    # # print(np.fft.ifft2(np.fft.fft2(a)))
    # # IDFT2(DFT2(image))
    # arr = np.array([1,2,3,4,5,6])
    # arr = arr[:,np.newaxis]
    # # ne_image =(np.fft.fft2(image))
    # # erez_image = (DFT2(image))
    #
    #
    # print(np.allclose(DFT2(image),np.fft.fft2(image)))
    # # print(DFT((DFT(image).transpose()).transpose()))
    #
    # # print("tetett")


    # print(np.fft.fft2(image))
    # print(ne_image)
    # print("new table")
    # print(erez_image)
    # plt.imshow(ne_image ,cmap=plt.get_cmap('gray'))
    # plt.show()

    # print(np.allclose(np.fft.ifft2(np.fft.fft2(image)),IDFT2(DFT2(image))))
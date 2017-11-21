import numpy as np
from scipy.misc import imread as imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


RGB2YIQ_MAT = np.array([[0.299, 0.587, 0.114],
               [0.596, -0.275, -0.321],
               [0.212, -0.523, 0.311]])

YIQ2RGB_MAT = np.linalg.inv(RGB2YIQ_MAT)


def read_image(filename, representation):
    im = imread(filename)
    if representation == 1:
        im = rgb2gray(im)

    elif representation == 2:
        im = (im / 255).astype(np.float64)

    return im


def imdisplay(filename, representation):
    img = read_image(filename, representation)
    plt.figure()
    if representation == 1:
        plt.imshow(img, cmap=plt.cm.gray)
    elif representation == 2:
        plt.imshow(img)
    plt.show()


def rgb2yiq(imRGB):
    imYIQ = np.zeros(imRGB.shape)

    r, g, b = imRGB[:, :, 0], imRGB[:, :, 1], imRGB[:, :, 2]
    imYIQ[:, :, 0] = RGB2YIQ_MAT[0, 0] * r + RGB2YIQ_MAT[0, 1] * g + RGB2YIQ_MAT[0, 2] * b
    imYIQ[:, :, 1] = RGB2YIQ_MAT[1, 0] * r + RGB2YIQ_MAT[1, 1] * g + RGB2YIQ_MAT[1, 2] * b
    imYIQ[:, :, 2] = RGB2YIQ_MAT[2, 0] * r + RGB2YIQ_MAT[2, 1] * g + RGB2YIQ_MAT[2, 2] * b

    return imYIQ


def yiq2rgb(imYIQ):
    imRGB = np.zeros(imYIQ.shape)

    y, i, q = imYIQ[:, :, 0], imYIQ[:, :, 1], imYIQ[:, :, 2]
    imRGB[:, :, 0] = YIQ2RGB_MAT[0, 0] * y + YIQ2RGB_MAT[0, 1] * i + YIQ2RGB_MAT[0, 2] * q
    imRGB[:, :, 1] = YIQ2RGB_MAT[1, 0] * y + YIQ2RGB_MAT[1, 1] * i + YIQ2RGB_MAT[1, 2] * q
    imRGB[:, :, 2] = YIQ2RGB_MAT[2, 0] * y + YIQ2RGB_MAT[2, 1] * i + YIQ2RGB_MAT[2, 2] * q

    return imRGB


def histogram_equalize_gray(img):
    round_img = (img*255).astype(np.uint8)
    hist_orig, bins = np.histogram(round_img, 256)
    # calculate the cumulative histogram and normalized it
    cumsum_hist = np.cumsum(hist_orig)
    cumsum_hisשt_norm = (cumsum_hist / cumsum_hist[-1])*255

    # make the linear stretch and create the output image
    im_eq = np.interp(round_img, bins[:-1], cumsum_hist_norm)
    im_eq = im_eq.reshape(img.shape)
    hist_eq = np.histogram(im_eq.astype(np.uint8), 256)[0]

    return (im_eq/255).astype(np.float64), hist_orig, hist_eq


def histogram_equalize(im_orig):

    if len(im_orig.shape) == 2:  # gray
        return histogram_equalize_gray(im_orig.copy())

    elif len(im_orig.shape) == 3:  # rgb
        yiq_im = rgb2yiq(im_orig.copy())
        y_channel, hist_orig, hist_eq = histogram_equalize_gray(yiq_im[:, :, 0].copy())
        yiq_im[:, :, 0] = y_channel

        return yiq2rgb(yiq_im), hist_orig, hist_eq


def quantize_gray(im_orig, n_quant, n_iter):
    # initializations:
    im_quant = im_orig
    error = []
    im_quant *= 255
    hist, bins = np.histogram(im_quant, 256)
    cum_hist = np.cumsum(hist)

    # initial z's:
    r, c = im_orig.shape
    znew = np.empty((n_quant + 1,)).astype(np.int)
    for i in range(n_quant):
        curr_min = np.where(cum_hist > (i * r * c / n_quant))
        znew[i] = curr_min[0][0]
    znew[0], znew[n_quant] = 0, 255

    hist_times_n = np.fromfunction(lambda j: j * hist[j], (256,), dtype=int)

    # finding the min:
    for i in range(n_iter):

        # calculate q:
        qnew = np.empty((n_quant,)).astype(np.int)
        for q in range(n_quant):
            sum_n = np.sum(hist_times_n[znew[q]:znew[q + 1] + 1])
            sum_h = np.sum(hist[znew[q]:znew[q + 1] + 1])
            qnew[q] = (round(sum_n / sum_h))

        # calculate z:
        prev_z = np.copy(znew)
        for z in range(1, n_quant):
            znew[z] = int((qnew[z - 1] + qnew[z]) / 2)

        sum_error = 0
        for e in range(n_quant):
            s = np.fromfunction(lambda j: ((qnew[e] - (j + znew[e])) * (qnew[e] - (j + znew[e]))) * hist[(j + znew[e])],
                                (znew[e + 1] - znew[e] - 1,), dtype=int)
            sum_error += np.sum(s)
        error.append(sum_error)

        # checking for convergence
        if np.array_equal(znew, prev_z):
            break

    # final image:
    quan_int = np.array([qnew[0]])
    for i in range(n_quant):
        quan_int = np.append(quan_int, np.full((znew[i + 1] - znew[i],), qnew[i], dtype=int))
    im_quant = np.interp(im_quant, np.arange(256), quan_int)

    return (im_quant/255).astype(np.float64), error


def quantize(im_orig, n_quant, n_iter):
    if len(im_orig.shape) == 2:  # gray
        return quantize_gray(im_orig.copy(), n_quant, n_iter)

    elif len(im_orig.shape) == 3:  # rgb
        yiq_im = rgb2yiq(im_orig.copy())
        y_channel, error = quantize_gray(yiq_im[:, :, 0], n_quant, n_iter)
        yiq_im[:, :, 0] = y_channel

        return yiq2rgb(yiq_im), error



if __name__ == '__main__':
    image, errors = quantize(read_image('F:\My Documents\Google Drive\תואר ראשון מדמח\שנה ג\עיבוד '
                                        'תמונה\exs\ex1\quantization_examples\\rgb_orig.png', 2), 3, 100)

    plt.imshow(image)
    plt.show()
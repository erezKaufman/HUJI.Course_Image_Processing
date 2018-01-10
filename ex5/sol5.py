from keras.layers import Input, Dense
from keras.models import Model
import os
import numpy as np
from scipy.ndimage.filters import convolve as convolve
from skimage.color import rgb2gray
from scipy.misc import imread
from scipy import signal

PIXEL_MAX_INTENSITY = 255


# a = Input(shape=(784,))
#
# x = Dense(64)(a)
# x = Dense(64)(x)
#
# predictions = Dense(10)(x)
#  in the end of the process of Model, i can do 'save' on that model, with the directory that I wish to save in to
#  later when I want to load, just do 'load'
# recomend that thegenerator function works.
# the generator is infinite! it will create every amount of pictures that the algorithm wishes to
#

# data_generator = load_dataset(filenames, batch_size, corruption_func, crop_size)


cached_images = {}
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


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    the method outputs data_generator, a Python’s generator object which outputs random tuples of the form
    (source_batch, target_batch), where each output variable is an array of shape (batch_size, 1,
    height, width), target_batch is made of clean images, and source_batch is their respective randomly
    corrupted version according to corruption_func(im). Each image in the batch should be picked
    at random from the given list of filenames, loaded as a grayscale image in the [0,1] range (you should
    use read_image from previous assignments), followed by corrupting the entire image with corruption_func(im),
    and finally randomly choosing the location of a patch the size of crop_size, and subtract
    the value 0.5 from each pixel in both the source and target image patches.
    :param filenames: A list of filenames of clean images.
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent.
    :param corruption_func: A function receiving a numpy’s array representation of an image as a single argument
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract.
    :return: data_generator
    """
    while True:
        random_images_number = np.random.randint(0, len(filenames), batch_size)  # get random image number
        source_batch = []
        target_batch = []
        for i in range(batch_size):
            # TODO - 10/01/18 maybe consider using @lru cache?
            if filenames[random_images_number[i]] in cached_images:
                clean_image = cached_images[filenames[random_images_number[i]]]
            else:
                cached_images[filenames[random_images_number[i]]] = read_image(filenames[random_images_number[i]], 1)
                clean_image = cached_images[read_image(filenames[random_images_number[i]], 1)]
            # clean_image = read_image(filenames[random_images_number[i]], 1)  # read image and get gray scale image
            corrupted_image = corruption_func(clean_image)  # get corrupted image

            # TODO - 10/01/18 check if the way I choose the segments is valid
            # height_number_of_seghments = clean_image.shape[0] // crop_size[0] # divide the image height into segments
            # width_number_of_segments = clean_image.shape[1] // crop_size[1] # divide the image width into segments

            random_height_segemnt = np.random.randint(0, clean_image.shape[0] - crop_size[0], 1)
            random_width_segemnt = np.random.randint(0, clean_image.shape[1] - crop_size[1], 1)
            # get the new sliced clean image
            cropped_clean_image = clean_image[random_height_segemnt:random_height_segemnt + crop_size[0],
                                  random_width_segemnt:random_width_segemnt + crop_size[1]]
            # get the new sliced corrupted image
            cropped_corrupted_image = corrupted_image[random_height_segemnt:random_height_segemnt + crop_size[0],
                                      random_width_segemnt:random_width_segemnt + crop_size[1]]
            cropped_clean_image -= 0.5
            cropped_corrupted_image -= 0.5
            target_batch.append(cropped_clean_image)
            source_batch.append(cropped_corrupted_image)
        yield (source_batch,target_batch)



def resblock(input_tensor, num_channels):
    """
    The function takes as input a symbolic input tensor and the number of channels for each of its
    convolutional layers, and returns the symbolic output tensor of the layer configuration described above.
    The convolutional layers should use “same” border mode, so as to not decrease the spatial dimension of
    the output tensor.
    :param input_tensor: symbolic input tensor
    :param num_channels: number of channels for wach of the convolutional layers
    :return:
    """
    pass


def build_nn_model(height, width, num_channels, num_res_blocks):
    """

    :param height:
    :param width:
    :param num_channels:
    :param num_res_blocks:
    :return:
    """
    pass


def train_model(model, images, corruption_func, batch_size, samples_per_epoch, num_epochs, num_valid_samples):
    """

    :param model:
    :param images:
    :param corruption_func:
    :param batch_size:
    :param samples_per_epoch:
    :param num_epochs:
    :param num_valid_samples:
    :return:
    """
    pass


def restore_image(corrupted_image, base_model):
    """

    :param corrupted_image:
    :param base_model:
    :return:
    """
    pass


def add_gaussian_noise(image, min_sigma, max_sigma):
    """

    :param image:
    :param min_sigma:
    :param max_sigma:
    :return:
    """
    pass


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """

    :param num_res_blocks:
    :param quick_mode:
    :return:
    """
    pass


def add_motion_blur(image, kernel_size, angle):
    """

    :param image:
    :param kernel_size:
    :param angle:
    :return:
    """
    pass


def random_motion_blur(image, list_of_kernel_sizes):
    """

    :param image:
    :param list_of_kernel_sizes:
    :return:
    """
    pass


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """

    :param num_res_blocks:
    :param quick_mode:
    :return:
    """
    pass

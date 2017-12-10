# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged 

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy import signal
from scipy.ndimage import label, center_of_mass

# TODO Change this to 'import sol4_utils'
from ex4.sol4_utils import read_image,build_gaussian_pyramid, blur_spatial,pyramid_blending

K = 0.04
X_DERIVED_FILTER = np.array([1,0,-1]).reshape(3,1)
Y_DERIVED_FILTER = X_DERIVED_FILTER.T

# TODO DELETE ME
def showIm(ims):
    """

    :param im:
    :return:
    """


    plt.imshow(ims,cmap=plt.get_cmap('gray'))
    plt.show()
#### DELETE ME

def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    i_x = convolve(im,X_DERIVED_FILTER) # TODO check if need to multiply filter by intensity (0.5)
    i_y = convolve(im,Y_DERIVED_FILTER)
    i_x_i_y = convolve(i_x,Y_DERIVED_FILTER)
    # i_y_i_x = convolve(i_y,X_DERIVED_FILTER)
    i_x_2 = convolve(i_x,X_DERIVED_FILTER)
    i_y_2 = convolve(i_y,Y_DERIVED_FILTER)
    blured_ix_iy = blur_spatial(i_x_i_y,3)
    blured_ix_2 = blur_spatial(i_x_2,3)
    blured_iy_2 = blur_spatial(i_y_2,3)
    row, col = im.shape
    matrix_image = np.zeros((row,col,2,2))
    matrix_image[:,:,0,0] = blured_ix_2
    matrix_image[:,:,0,1] = blured_ix_iy
    matrix_image[:,:,1,0] = blured_ix_iy
    matrix_image[:,:,1,1] = blured_iy_2

    first_product = np.linalg.det(matrix_image[:, :])
    trace_matrix = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            trace_matrix = matrix_image[i,j].trace()


    second_product = K * (trace_matrix) ** 2
    R = first_product - second_product
    # print(R)
    binary_image = non_maximum_suppression(R)
    # print(binary_image)
    return np.argwhere(binary_image==True)

    # plt.imshow(R,cmap=plt.get_cmap('gray'))
    # plt.show()
def sample_descriptor(im, pos, desc_rad):
    """
  Samples descriptors at the given corners.
  :param im: A 2D array representing an image.
  :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
  :param desc_rad: "Radius" of descriptors to compute.
  :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
  """
    pass


def find_features(pyr):
    """
  Detects and extracts feature points from a pyramid.
  :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
  :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
  """
    pass


def match_features(desc1, desc2, min_score):
    """
  Return indices of matching descriptors.
  :param desc1: A feature descriptor array with shape (N1,K,K).
  :param desc2: A feature descriptor array with shape (N2,K,K).
  :param min_score: Minimal match score.
  :return: A list containing:
              1) An array with shape (M,) and dtype int of matching indices in desc1.
              2) An array with shape (M,) and dtype int of matching indices in desc2.
  """
    pass


def apply_homography(pos1, H12):
    """
  Apply homography to inhomogenous points.
  :param pos1: An array with shape (N,2) of [x,y] point coordinates.
  :param H12: A 3x3 homography matrix.
  :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
  """
    pass


def ransac_homography(points1, points2, num_iter, inlier_tol):
    """
  Computes homography between two sets of points using RANSAC.
  :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
  :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
  :param num_iter: Number of RANSAC iterations to perform.
  :param inlier_tol: inlier tolerance threshold.
  :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal set of inlier matches found.
  """
    pass


def display_matches(im1, im2, points1, points2, inliers):
    """
  Dispalay matching points.
  :param im1: A grayscale image.
  :param im2: A grayscale image.
  :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
  :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
  :param inliers: An array with shape (S,) of inlier matches.
  """
    pass


def accumulate_homographies(H_succesive, m):
    """
  Convert a list of succesive homographies to a
  list of homographies to a common reference frame.
  :param H_successive: A list of M-1 3x3 homography
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
  :param m: Index of the coordinate system towards which we would like to
    accumulate the given homographies.
  :return: A list of M 3x3 homography matrices,
    where H2m[i] transforms points from coordinate system i to coordinate system m
  """
    pass


def render_panorama(ims, Hs):
    """
  Renders a panorama.
  :param ims: A list of grayscale images.
  :param Hs: A list of 3x3 homography matrices. Hs[i] is a homography which transforms
   points from the coordinate system of ims[i] to the coordinate system of the panorama.
  :return: A grayscale panorama image.
  """
    pass


def least_squares_homography(points1, points2):
    """
  Computes homography transforming points1 towards points2, using least squares method.
  points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
  :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
  :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
  :return: A 3X3 array with the computed homography. In case of instable solutions returns None.
  """
    p1, p2 = points1, points2
    o0, o1 = np.zeros((p1.shape[0], 1)), np.ones((p1.shape[0], 1))

    A = np.vstack([np.hstack([p1[:, :1], o0, -p1[:, :1] * p2[:, :1], p1[:, 1:], o0, -p1[:, 1:] * p2[:, :1], o1, o0]),
                   np.hstack([o0, p1[:, :1], -p1[:, :1] * p2[:, 1:], o0, p1[:, 1:], -p1[:, 1:] * p2[:, 1:], o0, o1])])

    # Return None for unstable solutions
    if np.linalg.matrix_rank(A, 1e-3) < 8:
        return None
    if A.shape[0] == 8 and np.linalg.cond(A) > 1e10:
        return None

    H = np.linalg.lstsq(A, p2.T.flatten())[0]
    H = np.r_[H, 1]
    return H.reshape((3, 3)).T


def non_maximum_suppression(image):
    """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image, where True indicates local maximum.
  """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
  Splits the image im to m by n rectangles and uses harris_corner_detector on each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary of the image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramaGenerator:
    """
  Generates panorama from a set of images.
  """

    def __init__(self, data_dir, file_prefix, num_images):
        """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 1,2,..
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panorama with.
    """
        self.files = [os.path.join(data_dir, '%s%d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]

        # Read images.
        self.images = [read_image(f, 1) for f in self.files]
        self.panorama = None

    def generate_panorama(self):
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for im in self.images:
            pyramid, _ = build_gaussian_pyramid(im, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 10000, 6)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should not display any figures!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the panorama coordinate system.
        Htot = accumulate_homographies(Hs, (len(Hs) - 1) // 2)

        # Final panorama is generated using 3 channels of the RGB images
        images_rgb = [read_image(f, 2) for f in self.files]

        # Render panorama for each color channel and combine them.
        panorama = []
        for channel in range(3):
            channel_images = [im[..., channel] for im in images_rgb]
            channel_panorama = render_panorama(channel_images, Htot)
            panorama.append(channel_panorama)
        self.panorama = np.dstack(panorama)
        return self.panorama

    def show_panorama(self, figsize=(20, 20)):
        assert self.panorama is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panorama.clip(0, 1))
        plt.show()
if __name__ == '__main__':
    image1 = read_image("/cs/usr/erez/Documents/image processing/exs/HUJI.Course_Image_Processing/ex4/ex4-impr-master/external/office1.jpg",1)
    image2 = read_image("/cs/usr/erez/Documents/image processing/exs/HUJI.Course_Image_Processing/ex4/ex4-impr-master/external/office2.jpg",1)
    image3 = read_image("/cs/usr/erez/Documents/image processing/exs/HUJI.Course_Image_Processing/ex4/ex4-impr-master/external/office3.jpg",1)
    image4 = read_image("/cs/usr/erez/Documents/image processing/exs/HUJI.Course_Image_Processing/ex4/ex4-impr-master/external/office4.jpg",1)
    harris_corner_detector(image1)
    harris_corner_detector(image2)
    harris_corner_detector(image3)
    harris_corner_detector(image4)
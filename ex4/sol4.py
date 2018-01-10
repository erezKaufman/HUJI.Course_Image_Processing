# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged
import shutil

import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import accumulate
from scipy.ndimage.filters import convolve
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import map_coordinates
from scipy.ndimage import label, center_of_mass

# TODO Change this to 'import sol4_utils'
import ex4.sol4_utils as sol4_utils

K = 0.04
X_DERIVED_FILTER = np.array([1, 0, -1]).reshape(3, 1)
Y_DERIVED_FILTER = X_DERIVED_FILTER.T


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """

    # TODO check if need to multiply filter by intensity (0.5)
    i_x = convolve(im, X_DERIVED_FILTER)

    i_y = convolve(im, Y_DERIVED_FILTER)

    i_x_i_y = i_x * i_y

    i_x_2 = i_x * i_x
    i_y_2 = i_y * i_y
    blured_ix_iy = sol4_utils.blur_spatial(i_x_i_y, kernel_size=3)
    blured_ix_2 = sol4_utils.blur_spatial(i_x_2, kernel_size=3)
    blured_iy_2 = sol4_utils.blur_spatial(i_y_2, kernel_size=3)
    row, col = im.shape
    matrix_image = np.zeros((row, col, 2, 2))  # TODO can be better if I save the process of creating the matrix_image
    matrix_image[:, :, 0, 0] = blured_ix_2
    matrix_image[:, :, 0, 1] = blured_ix_iy
    matrix_image[:, :, 1, 0] = blured_ix_iy
    matrix_image[:, :, 1, 1] = blured_iy_2

    first_product = np.linalg.det(matrix_image)
    trace = matrix_image[:, :, 0, 0] + matrix_image[:, :, 1, 1]

    second_product = K * (trace ** 2)
    R = first_product - second_product
    binary_image = non_maximum_suppression(R)

    max_coordinates = np.argwhere(binary_image == True)
    returned_val = np.fliplr(max_coordinates)
    return returned_val


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    pos = pos / 4

    coord_results = []
    dummy_matrix = np.zeros((desc_rad * 2 + 1, desc_rad * 2 + 1))
    for y_coord, x_coord in pos:
        cur_x_coord_range = np.arange(x_coord - desc_rad, x_coord + desc_rad + 1)
        cur_y_coord_range = np.arange(y_coord - desc_rad, y_coord + desc_rad + 1)
        grid = np.meshgrid(cur_x_coord_range, cur_y_coord_range)
        cur_map_coord = map_coordinates(im, grid, order=1, prefilter=False)
        cur_map_mean = np.mean(cur_map_coord)
        if np.count_nonzero((cur_map_coord - cur_map_mean)) == 0:
            result_descriptor = dummy_matrix.copy()
        else:
            result_descriptor = (cur_map_coord - cur_map_mean) / np.linalg.norm(cur_map_coord - cur_map_mean)

        coord_results.append(result_descriptor)
    test = np.array(coord_results)
    return test


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
    """
    original_image = pyr[0]
    third_level_image = pyr[2]
    return_coord_list = spread_out_corners(original_image, 7, 7, 3)
    returned_descriptors = sample_descriptor(third_level_image, return_coord_list, 3)
    return [return_coord_list, returned_descriptors]


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
    # flat each descriptor matrix to be shape(Ni,desc_rad**2)
    flattend_desc1 = desc1.reshape(desc1.shape[0], desc1.shape[1] ** 2)
    flattend_desc2 = desc2.reshape(desc2.shape[0], desc2.shape[1] ** 2).T

    # create score matrix for each descriptor with other descriptors of the second image
    S_matrix = np.dot(flattend_desc1, flattend_desc2)

    # create sorted matrix of the score matrix by col and row
    s_sorted_by_row = np.sort(S_matrix.copy(), axis=1)
    s_sorted_by_col = np.sort(S_matrix.copy(), axis=0)

    result = np.all(((S_matrix >= s_sorted_by_col[-2]), (S_matrix.T >= s_sorted_by_row[:, -2]).T, (S_matrix >
                                                                                                   min_score)), axis=0)
    returned_arrays = np.where(result)

    return returned_arrays


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    pos1 = np.insert(pos1, pos1.shape[1], 1, axis=1)
    temp_homography1 = np.dot(H12, pos1.T).T
    z = temp_homography1[:, 2].reshape(temp_homography1.shape[0], 1)
    returned_array = temp_homography1[:, 0:2] / z
    return returned_array


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    def ransac_homography_step_1(points1, points2, translation_only):
        """
        takes two random indices for matched points in im1 and im2, and returns an array for them
        :param points1:
        :param points2:
        :param rand_points_picked:
        :param translation_only:
        :return:
        """

        # Pick a random set of 2 point matches from pos1 and pos2.  We call these two sets of 2 points in the two
        # images P1J and P2J
        rand_matched_points = np.random.randint(0, points1.shape[0], 2)

        if translation_only:
            p1j_helper = np.array([points1[rand_matched_points[0]]])
            p2j_helper = np.array([points2[rand_matched_points[0]]])
        else:
            p1j_helper = np.array([points1[rand_matched_points[0]], points1[rand_matched_points[1]]])
            p2j_helper = np.array([points2[rand_matched_points[0]], points2[rand_matched_points[1]]])

        return p1j_helper, p2j_helper

    def ransac_homogrophy_step_3(h12_helper, inlier_tol_helper, largest_inlier_set_helper, p2j_helper, points1_helper):
        """
        the function is responsible to the step 3 of the ransac algorightm, on which we calculate all the scores of
        each pixel to the current homography, and check if it is the best solution for our images
        :param h12_helper:
        :param inlier_counter:
        :param inlier_tol_helper:
        :param largest_inlier_set_helper:
        :param outier_counter:
        :param p2j_helper:
        :param points1_helper:
        :return:
        """

        p2j_tag = apply_homography(points1_helper, h12_helper)
        ej_norm = np.sum(np.abs(np.asarray(p2j_tag - p2j_helper)) ** 2, axis=-1) ** (1. / 2)
        ej_final = ej_norm ** 2
        current_inlier_set = np.argwhere(ej_final < inlier_tol_helper)
        if current_inlier_set.size > largest_inlier_set_helper.size:
            largest_inlier_set_helper = current_inlier_set
        return largest_inlier_set_helper

    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """

    largest_inlier_set = np.array([])
    for i in range(num_iter):
        # --Step 1 --
        p1j, p2j = ransac_homography_step_1(points1, points2, translation_only)

        # --Step 2 --
        h12 = np.asarray(estimate_rigid_transform(p1j, p2j, translation_only))
        h12 = h12 / h12[2, 2]

        # --Step 3 --
        largest_inlier_set = ransac_homogrophy_step_3(h12, inlier_tol, largest_inlier_set, points2, points1)

    # recompute the homography over Jin
    p1j_in = points1[largest_inlier_set].reshape((len(largest_inlier_set), 2))
    p2j_in = points2[largest_inlier_set].reshape((len(largest_inlier_set), 2))
    returned_h = estimate_rigid_transform(p1j_in, p2j_in, translation_only)

    return returned_h, largest_inlier_set.reshape((largest_inlier_set.shape[0],))


def display_matches(im1, im2, points1, points2, inliers):  # TODO COMPLETE THIS
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    col = im1.shape[1]
    points2[:, 0] = points2[:, 0] + col
    horizontal_concat_image = np.hstack((im1, im2))
    # length = points1.shape[0]
    im1_inlier = points1[inliers].reshape(len(inliers), 2)
    im2_inlier = points2[inliers].reshape(len(inliers), 2)
    # print(im1_inlier.shape)

    # im1_outlier = np.delete(points1, inliers).reshape(length-len(inliers),2)
    # im2_outlier = np.delete(points2, inliers).reshape(length-len(inliers),2)


    # x_inliers[0] = im1_inlier[:, 0]
    # y_inliers[0] = im1_inlier[:, 1]
    # x_inliers[1] = im2_inlier[:, 0]
    # y_inliers[1] = im2_inlier[:, 1]
    x1 = im1_inlier[:, 0]  # x coordinates of all inliers for image one
    y1 = im1_inlier[:, 1]  # y coordinates of all inliers for image one
    x2 = im2_inlier[:, 0]  # x coordinates of all inliers for image two
    y2 = im2_inlier[:, 1]  # y coordinates of all inliers for image two
    # new empty list to contain all  x coordinates of matching dots in image1 and image2
    xs_list = np.zeros((2, points1.shape[0]))
    xs_list[0] = points1[:, 0]
    xs_list[1] = points2[:, 0]

    # new empty list to contain all  y coordinates of matching dots in image1 and image2
    ys_list = np.zeros((2, points1.shape[0]))
    ys_list[0] = points1[:, 1]
    ys_list[1] = points2[:, 1]

    # print(x1.shape)
    # print(points1.shape)
    # print(points2.shape)
    # print(points1[:,1].shape)
    # all_points_xs = [points1[:, 0], points2[:, 0]]
    #
    # all_points_ys = [points1[:, 1], [points2[:, 1]]]
    plt.scatter(xs_list, ys_list, c='r', s=5)
    plt.plot(xs_list, ys_list, color='b', linewidth=.4)
    plt.plot([x1, x2], [y1, y2], color='y', linewidth=.4)
    plt.imshow(horizontal_concat_image, cmap=plt.get_cmap('gray'))
    #
    # plt.figure()
    # plt.imshow(horizontal_concat_image, cmap=plt.get_cmap('gray'))
    # plt.plot(x1, y1, '.', x2, y2, '.')
    # x = res2[:, 0]
    # y = res2[:, 1]
    # plt.figure()
    # plt.imshow(image4)
    # plt.plot(x, y, '.')
    plt.show()
    # print(inliers)


# def display_matches(im1, im2, points1, points2, inliers):
#     """
#     Dispalay matching points.
#
#     :param im1: A grayscale image.
#     :param im2: A grayscale image.
#     :param points1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
#     :param points2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
#     :param inliers: An array with shape (S,) of inlier matches.
#     """
#     x_inliers = np.zeros((2, len(inliers)))
#     y_inliers = np.zeros((2, len(inliers)))
#     combined_images = np.hstack((im1, im2))
#
#     im1_inliers = points1[inliers, :]
#     im2_inliers = points2[inliers, :]
#
#     x_inliers[0] = im1_inliers[:, 0]
#     x_inliers[1] = im2_inliers[:, 0] + im1.shape[1]
#     y_inliers[0] = im1_inliers[:, 1]
#     y_inliers[1] = im2_inliers[:, 1]
#
#     plt.plot(x_inliers, y_inliers, mfc='r', c='b', lw=.4, ms=5, marker='o', color='yellow')
#     plt.imshow(combined_images, cmap="gray")
#     plt.show()

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
    H_succesive = np.asarray(H_succesive)
    norm_array = H_succesive[:, 2, 2][:, None]
    h_length = len(H_succesive)
    H_succesive = H_succesive / norm_array[:, None]
    returned_list = [0] * (h_length + 1)
    new_reversed_list = H_succesive[:m]
    new_reversed_list = new_reversed_list[::-1]

    returned_list[:m] = list(accumulate(new_reversed_list, func=lambda x, y: x.dot(y)))
    returned_list[m] = np.eye(3)
    # new_reversed_list = H_succesive[m:]
    # new_reversed_list = new_reversed_list[::-1]
    # TODO - considet maybe in the following line I need to run the list backwards
    returned_list[m + 1:] = list(accumulate(np.linalg.inv(H_succesive[m:]), func=lambda x, y: x.dot(y)))
    # norm_array = returned_list[:, 2, 2][:, None]
    # returned_list= returned_list/ norm_array[:, None] # TODO need to normilize this!!

    return returned_list


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """

    four_coords = np.zeros((4, 2))
    four_coords[0] = [0, 0]
    four_coords[1] = [0, h-1]
    four_coords[2] = [w-1, 0]
    four_coords[3] = [w-1, h-1]

    new_cords = apply_homography(four_coords, homography)
    top_left = [new_cords[:, 0].min().round().round(), new_cords[:, 1].min().round()]
    bottom_right = [new_cords[:, 0].max().round(), new_cords[:, 1].max().round()]
    return np.asarray([top_left, bottom_right]).astype(np.int)


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    h, w = image.shape
    top_left, bottom_right = compute_bounding_box(homography, w, h)
    # top_left, bottom_right = compute_bounding_box(homography, w, h)

    # returned_image = np.zeros((bottom_right[0]-top_left[0],bottom_right[1]-top_left[1]))
    cur_y_coord_range = np.arange(top_left[1], bottom_right[1])
    cur_x_coord_range = np.arange(top_left[0], bottom_right[0])
    grid = np.meshgrid(cur_x_coord_range, cur_y_coord_range)
    grid = np.asarray([grid[0].flatten(), grid[1].flatten()]).T
    image_in_reference_coords = np.fliplr(apply_homography(grid, np.linalg.inv(homography))).T
    cur_map_coord = map_coordinates(image, image_in_reference_coords, order=1, prefilter=False)

    returned_image = np.asarray(cur_map_coord).reshape((bottom_right[1] - top_left[1], bottom_right[0] - top_left[0]))
    return returned_image


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    returned_image = np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])
    plt.imshow(returned_image)
    plt.show()
    return returned_image


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


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


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
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
            H12, inliers = ransac_homography(points1, points2, 100, 50, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            plt.imsave('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()


def show_harris():
    image1 = sol4_utils.read_image(relpath("ex4-impr-master\external\oxford1.jpg"), 1)
    image12 = sol4_utils.read_image(relpath("panorama_test1.jpg"), 1)
    image14 = sol4_utils.read_image(relpath("panorama_test1.jpg"), 2)
    image11 = sol4_utils.read_image(relpath("panorama_test2.jpg"), 1)
    image13 = sol4_utils.read_image(relpath("panorama_test2.jpg"), 2)
    image2 = sol4_utils.read_image(relpath("ex4-impr-master\external\oxford1.jpg"), 2)
    image3 = sol4_utils.read_image(relpath("ex4-impr-master\external\oxford2.jpg"), 1)
    image4 = sol4_utils.read_image(relpath("ex4-impr-master\external\oxford2.jpg"), 2)
    image5 = sol4_utils.read_image(relpath("ex4-impr-master\external\\backyard2.jpg"), 1)
    image6 = sol4_utils.read_image(relpath("ex4-impr-master\external\\backyard2.jpg"), 2)
    image7 = sol4_utils.read_image(relpath("ex4-impr-master\external\\backyard3.jpg"), 1)
    image8 = sol4_utils.read_image(relpath("ex4-impr-master\external\\backyard3.jpg"), 2)
    res1 = spread_out_corners(image1, 7, 7, 3)
    res2 = spread_out_corners(image3, 7, 7, 3)
    # sample_descriptor(image1, res1, 3)
    x = res1[:, 0]
    y = res1[:, 1]
    plt.figure()
    # plt.imshow(image11)
    plt.imshow(image1, cmap=plt.get_cmap('gray'))
    #
    plt.plot(x, y, '.')
    #
    x = res2[:, 0]
    y = res2[:, 1]
    plt.figure()
    # plt.imshow(image12)
    plt.imshow(image3, cmap=plt.get_cmap('gray'))
    plt.plot(x, y, '.')
    plt.show()


def compute_ransac_and_display():
    image1 = sol4_utils.read_image(relpath("ex4-impr-master\external\oxford1.jpg"), 1)
    # image12 = sol4_utils.read_image(relpath("ex4-impr-master\external\DSC_0784.JPG"), 1)
    # image14 = sol4_utils.read_image(relpath("ex4-impr-master\external\DSC_0784.JPG"), 2)
    # image11 = sol4_utils.read_image(relpath("ex4-impr-master\external\DSC_0783.JPG"), 1)
    # image13 = sol4_utils.read_image(relpath("ex4-impr-master\external\DSC_0783.JPG"), 2)
    image2 = sol4_utils.read_image(relpath("ex4-impr-master\external\oxford1.jpg"), 2)
    image3 = sol4_utils.read_image(relpath("ex4-impr-master\external\oxford2.jpg"), 1)
    image4 = sol4_utils.read_image(relpath("ex4-impr-master\external\oxford2.jpg"), 2)
    image5 = sol4_utils.read_image(relpath("ex4-impr-master\external\\backyard1.jpg"), 1)
    image6 = sol4_utils.read_image(relpath("ex4-impr-master\external\\backyard1.jpg"), 2)
    image7 = sol4_utils.read_image(relpath("ex4-impr-master\external\\backyard2.jpg"), 1)
    image8 = sol4_utils.read_image(relpath("ex4-impr-master\external\\backyard2.jpg"), 2)

    gpyr1 = sol4_utils.build_gaussian_pyramid(image1, 3, 3)[0]
    gpyr2 = sol4_utils.build_gaussian_pyramid(image3, 3, 3)[0]
    points1, desc1 = find_features(gpyr1)
    points2, desc2 = find_features(gpyr2)
    # print(points1)
    ind1, ind2 = match_features(desc1, desc2, 0.5)
    points1, points2 = points1[ind1, :], points2[ind2, :]

    H12, inliers = ransac_homography(points1, points2, 1000, 10)

    # warp_image(image6,
    # display_matches(image5, image7, points1, points2, inliers)
    # H = accumulate_homographies()
    H11 = np.eye(3)
    # H11[1,1] = 2
    warped_channel = warp_channel(image1, H11)
    plt.imshow(warped_channel, cmap='gray')
    plt.show()


if __name__ == '__main__':
    # I_matrix = np.eye(3)
    # I1 = np.array([1,0,0,0,2,0,0,0,1]).reshape(3,3)
    # I21 = np.array([2,0,0,0,0.5,0,0,0,1]).reshape(3,3)
    # I3 = I_matrix*2
    # accumulate_homographies([I1], 0)
    # compute_ransac_and_display()

    p = PanoramicVideoGenerator(os.getcwd(), 'oxford', 2)
    p.align_images(False)
    p.generate_panoramic_images(1)
    p.show_panorama(0)
    # show_harris()

    # a = np.arange(10 * 4).reshape(10, 2, 2)
    # b = np.arange(12 * 4).reshape(12, 2, 2)
    # ind1, ind2 = match_features(a, b, 0.5)
    # a, b = a[ind1], b[ind2]
    # print(a, b)

    # b = np.array([1, 2, 3, 4, 5, 6]).reshape(3, 2)  # 1*2+2*3+3*4= 2+6+12=20
    # b = np.insert(b, 2, 1, axis=1)
    # print(b)
    # print(b.shape)
    # d = np.array([5, 6]).reshape(2, 1)  # 1*2+2*3+3*4= 2+6+12=20
    # y = np.array([1, 1]).reshape(2, 1)  # 1*2+2*3+3*4= 2+6+12=20
    # vecs = np.array([b, d,y])
    # print(np.insert(vecs,2,1,axis=-1))
    # c = np.dot(a, b)
    # print(c)
    # print(a)
    # print(np.sort(a,axis=1)) # by row
    # print(np.sort(a,axis=0)) # by col

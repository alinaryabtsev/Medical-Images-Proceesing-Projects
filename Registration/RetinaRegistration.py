import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import transform as tf
from skimage import segmentation, filters, exposure, restoration, registration
import utils

REGISTRATION_BY_FEATURES = 1
REGISTRATION_BY_SEGMENTATION = 2

BLACK_LINE = 100
GOOD_MATCH_PERCENT = 0.2
REGISTRATION_MUM_OF_POINTS = 4


class RegistrationException(Exception):
    pass


class RetinaRegistration:
    def __init__(self, bl_filename, fu_filename, registation_type):
        self.bl_image = cv2.imread(bl_filename, 0)[:-BLACK_LINE, :]
        self.fu_image = cv2.imread(fu_filename, 0)[:-BLACK_LINE, :]
        self.registration_type = registation_type
        if registation_type == REGISTRATION_BY_FEATURES:
            self.key_points_bl, self.key_points_fu = None, None
            self.descriptors_bl, self.descriptors_fu = None, None
            self.points_bl, self.points_fu = None, None
        elif registation_type == REGISTRATION_BY_SEGMENTATION:
            self.bl_seg = None
            self.fu_seg = None

    def execute_registration(self, registration_type):
        if registration_type == REGISTRATION_BY_FEATURES:
            return self.compute_registrated_image_by_features()
        if registration_type == REGISTRATION_BY_SEGMENTATION:
            return self.compute_registrated_image_by_blood_vessels_segmentation()

    def find_retina_features(self, is_debug=False):
        """
        This function finds strong features in both BL and FU images to use for registration.
        :param is_debug: Shows graphs for debugging purposes if True.
        """
        orb = cv2.ORB_create()
        self.key_points_bl, self.descriptors_bl = orb.detectAndCompute(self.bl_image, None)
        self.key_points_fu, self.descriptors_fu = orb.detectAndCompute(self.fu_image, None)
        if is_debug:
            cv2.imshow("Key points of BL Image", cv2.drawKeypoints(self.bl_image, self.key_points_bl, None))
            cv2.imshow("Key points of FU Image", cv2.drawKeypoints(self.fu_image, self.key_points_fu, None))
            cv2.waitKey()

    def brute_force_matcher(self, is_debug=False):
        """
        This function performs brute-force on the BL and FU images.
        :param is_debug: Shows graphs for debugging purposes if True.
        """
        brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = brute_force.match(self.descriptors_bl, self.descriptors_fu)
        # finding the humming distance of the matches and sorting them
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:int(len(matches) * GOOD_MATCH_PERCENT)]
        if is_debug:
            output_image = cv2.drawMatches(self.bl_image, self.key_points_bl, self.fu_image,
                                           self.key_points_fu, matches, None, flags=2)
            cv2.imshow('Best matches', output_image)
            cv2.waitKey()
        self.points_bl = np.zeros((len(matches), 2), dtype=np.float32)
        self.points_fu = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            self.points_bl[i, :] = self.key_points_bl[match.queryIdx].pt
            self.points_fu[i, :] = self.key_points_fu[match.trainIdx].pt

    def calc_point_based_reg(self, bl_points, fu_points):
        """
        This function returns transformation matrix of the translation and rotation
        :return: a 3x3 rigidReg rigid 2D transformation matrix
        """
        t, r = self._optimal_rigid_transformation(bl_points, fu_points)
        return np.vstack((np.hstack((r, t)), np.array([0, 0, 1])))

    def _compute_weighted_entroids(self, bl_points, fu_points):
        """
        Computes weighted centroids
        :return: base line centroid and follow up centroid
        """
        return np.average(bl_points, axis=0), np.average(fu_points, axis=0)

    def _compute_centered_vectors(self, bl_points, fu_points):
        """
        Computes weighted vectors
        :return: base line and follow up centered vectors
        """
        bl_centorid, fu_centorid = self._compute_weighted_entroids(bl_points, fu_points)
        return bl_points - bl_centorid, fu_points - fu_centorid

    def _comute_covariance_matrix(self, bl_points, fu_points):
        """
        Computes covariance matrix of the centered vectors
        :return: 2x2 covariance matrix
        """
        x, y = self._compute_centered_vectors(bl_points, fu_points)
        return x.T @ np.eye(x.shape[0]) @ y

    def _optimal_rigid_transformation(self, bl_points, fu_points):
        """
        Computes optimal rigid transformation
        :return: a translation 2x1 vector and 2x2 roatation matrix transformaiton
        """
        u, s, v = np.linalg.svd(self._comute_covariance_matrix(bl_points, fu_points))
        d = np.eye(2)
        d[-1, -1] = np.linalg.det(v @ u.T)
        r = v.T @ d @ u
        p_centeroids, q_centeroids = self._compute_weighted_entroids(bl_points, fu_points)
        return (q_centeroids - r @ p_centeroids).reshape(-1, 1), r

    @staticmethod
    def calc_dist(x, y, rigid_reg):
        """
        This function computes the distance of each transformed point from its matching point in pixel units.
        :param rigid_reg: rigid registration matrix
        :return: a vector of length N which describes the RSME
        """
        return np.sqrt(np.mean(((rigid_reg @ np.vstack((x.T, np.ones(x.shape[0])))).T[:, :2] - y) ** 2, axis=1))

    def show_registration(self, warpped_image):
        """
        This function shows the registration.
        :param warpped_image: the warped image
        """
        plt.imshow(self.bl_image, cmap="gray")
        plt.imshow(warpped_image, alpha=0.5)
        plt.suptitle("Base line image and registrated image overlayed")
        plt.show()

    def compute_registrated_image_by_features(self):
        """
        Tnis function computes the resirated image and shows it.
        :return the images after registration: the baseline image and the warpped follow up image after regustration
        """
        self.find_retina_features(False)
        self.brute_force_matcher()
        iterations_number = 10000
        inlier_distatncce_th = 10
        inlier_th_ratio = 0.1
        reg, inliers_indicies = utils.ransac(self.points_bl, self.points_fu, self.calc_point_based_reg,
                                             RetinaRegistration.calc_dist, REGISTRATION_MUM_OF_POINTS,
                                             iterations_number, inlier_distatncce_th, inlier_th_ratio)
        if not np.any(reg):
            raise RegistrationException("returned rgistration from RANSAC is the zero transformation")
        transformed = tf.AffineTransform(reg)
        warpped_image = tf.warp(self.fu_image, transformed.params)
        self.show_registration(warpped_image)
        return self.bl_image, warpped_image

    def segment_blood_vessels(self):
        """
        This function creates segmentation images of the provided baseline and follow up images.
        """
        bl_denoised = restoration.denoise_bilateral(self.bl_image)
        bl_equalized = exposure.equalize_hist(bl_denoised)
        th_bl = filters.threshold_minimum(bl_equalized)
        self.bl_seg = bl_equalized > th_bl
        fu_denoised = restoration.denoise_bilateral(self.fu_image)
        fu_equalized = exposure.equalize_hist(fu_denoised)
        th_fu = filters.threshold_minimum(fu_equalized)
        self.fu_seg = fu_equalized > th_fu

    def compute_registrated_image_by_blood_vessels_segmentation(self):
        """
        This function computes regestration by blood vessels segmentation.
        :return: the images after registration: the baseline image and the warpped follow up image after regustration
        """
        self.segment_blood_vessels()
        best_shift, err, best_rotation = registration.phase_cross_correlation(self.bl_seg, self.fu_seg)
        best_shift = np.flip(-best_shift)
        transformed = tf.AffineTransform(rotation=best_rotation, translation=best_shift)
        warpped_image = tf.warp(self.fu_image, transformed.params)
        self.show_registration(warpped_image)
        return self.bl_image, warpped_image

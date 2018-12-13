#!/usr/bin/python
"""
Finds and highlights lane lines in dashboard camera videos.
See README.md for more info.

Author: Peter Moran
Created: 8/1/2017
"""
# from typing import List

import cv2
import numpy as np
from imageio.core import NeedDownloadError
from windows import Window, filter_window_list, joint_sliding_window_update, window_image


# try:
#    from moviepy.editor import VideoFileClip
# except NeedDownloadError as download_err:
#    if 'ffmpeg' in str(download_err):
#        prompt = input('The dependency `ffmpeg` is missing, would you like to download it? [y]/n')
#        if prompt == '' or prompt == 'y' or prompt == 'Y':
#            from imageio.plugins import ffmpeg
#
#            ffmpeg.download()
#            from moviepy.editor import VideoFileClip
#        else:
#            raise download_err
#    else:
#        # Unknown download error
#        raise download_err

# REGULATION_LANE_WIDTH = 4.2

class DashboardCamera(object):
    def __init__(self, scale_correction=(5.0 / 240, 2.0 / 240), h=240, w=320,
                 source=np.float32([(160 - 50, 0), (160 + 50, 0), ((320 - 570) / 2, 240), ((320 + 570) / 2, 240)]),
                 destination=np.float32([(20, 0), (300, 0), (20, 240), (300, 240)])):
        """
        Handles camera calibration, distortion correction, perspective warping, and maintains various camera properties.

        :param chessboard_img_fnames: List of file names of the chessboard calibration images.
        :param chessboard_size: Size of the calibration chessboard.
        :param lane_shape: Pixel locations of the four corners describing the profile of the lane lines on a straight
        road. Should be ordered clockwise, from the top left.
        :param scale_correction: Constants y_m_per_pix and x_m_per_pix describing the number of meters per pixel in the
        overhead transformation of the road.
        """
        # Get image size

        self.img_size = (h, w)
        self.img_height = h
        self.img_width = w

        # Define overhead transform and its inverse

        self.y_m_per_pix = scale_correction[0]
        self.x_m_per_pix = scale_correction[1]

        self.overhead_transform = cv2.getPerspectiveTransform(source, destination)
        self.inverse_overhead_transform = cv2.getPerspectiveTransform(destination, source)

    def warp_to_overhead(self, undistorted_img):
        """
        Transforms this camera's images from the dashboard perspective to an overhead perspective.

        Note: Make sure to undistort first.
        """
        return cv2.warpPerspective(undistorted_img, self.overhead_transform, dsize=(self.img_width, self.img_height))

    def warp_to_dashboard(self, overhead_img):
        """
        Transforms this camera's images from an overhead perspective back to the dashboard perspective.
        """
        return cv2.warpPerspective(overhead_img, self.inverse_overhead_transform,
                                   dsize=(self.img_width, self.img_height))


class LaneFinder(object):
    def __init__(self, cam, window_shape=(20, 61), search_margin=200, max_frozen_dur=15):
        self.camera = cam
        # Create windows
        self.windows_left = []
        self.windows_right = []
        for level in range(cam.img_height // window_shape[0]):  # 图像高度240,一个窗口高度20,一列有12个窗口，左右两列
            x_init_l = cam.img_width / 4
            x_init_r = cam.img_width / 4 * 3
            # 窗口序号从下往上逐渐增大
            self.windows_left.append(Window(level, window_shape, cam.img_size, x_init_l, max_frozen_dur))
            self.windows_right.append(Window(level, window_shape, cam.img_size, x_init_r, max_frozen_dur))
        self.search_margin = search_margin

    def find_lines(self, img_dash_undistorted):
        """
        Primary function for fitting lane lines in an image.

        Visualization options include:
        'dash_undistorted', 'overhead', 'lab_b', 'lab_b_binary', 'lightness', 'lightness_binary', 'value',
        'value_binary', 'pixel_scores', 'windows_raw', 'highlighted_lane', 'presentation'

        :param img_dashboard: Raw dashboard camera image taken by the calibrated `self.camera`.
        :param visuals: A list of visuals you would like to be saved to `self.visuals`.
        :return: A set of points along the left and right lane line: y_fit, x_fit_left, x_fit_right.
        """
        # ipm转换，转换矩阵需要经过标定得到 或者 手动微调看效果
        img_overhead = self.camera.warp_to_overhead(img_dash_undistorted)
        cv2.imwrite('./pic_watch/img_overhead.png', img_overhead)

        # pixel_scores = self.score_pixels(img_overhead)
        pixel_scores = img_overhead
        point_line = img_overhead
        # Select windows
        joint_sliding_window_update(self.windows_left, self.windows_right, pixel_scores, margin=self.search_margin)
        # Filter window positions
        win_left_valid, argvalid_l = filter_window_list(self.windows_left, remove_frozen=False, remove_dropped=True)
        win_right_valid, argvalid_r = filter_window_list(self.windows_right, remove_frozen=False, remove_dropped=True)

        # assert len(win_left_valid) >= 3 and len(win_right_valid) >= 3, 'Not enough valid windows to create a fit.'
        if len(win_left_valid) < 3 or len(win_right_valid) < 3:
            print('Not enough valid windows to create a fit.')
            return -1, -1, -1, -1, False
        # TODO: Do something if not enough windows to fit. Most likely fall back on old measurements.

        # Apply fit
        fit_vals = self.fit_lanes(zip(*[window.pos_xy() for window in win_left_valid]),
                                  zip(*[window.pos_xy() for window in win_right_valid]), point_line)

        # Find a safe region to apply the polynomial fit over. We don't want to extrapolate the shorter lane's extent.
        short_line_max_ndx = min(argvalid_l[-1], argvalid_r[-1])
        # Determine the location of the polynomial fit line for each row of the image
        y_fit = np.array(range(self.windows_left[short_line_max_ndx].y_begin, self.windows_left[0].y_end))
        x_fit_left = fit_vals['al'] * y_fit ** 2 + fit_vals['bl'] * y_fit + fit_vals['x0l']
        x_fit_right = fit_vals['ar'] * y_fit ** 2 + fit_vals['br'] * y_fit + fit_vals['x0r']

        fit_vals_curve = self.fit_curve(y_fit, (x_fit_left + x_fit_right) / 2.0)
        x_fit_center = fit_vals_curve['a'] * y_fit ** 2 + fit_vals_curve['b'] * y_fit + fit_vals_curve['x0']
        x_fit_center_derivative = 2.0 * fit_vals_curve['a'] * y_fit + fit_vals_curve['b']

        # Calculate radius of curvature
        # curve_radius = self.calc_curvature(win_left_valid)

        processed_frame_overhead = viz_lane(img_overhead, self.camera, x_fit_left, x_fit_right, y_fit, x_fit_center,
                                            "overhead")
        cv2.imwrite("./pic_watch/processed_frame_overhead.png", ~processed_frame_overhead)

        return y_fit, x_fit_center, x_fit_center_derivative, True

    def score_pixels(self, img):
        """
        Takes a road image and returns an image where pixel intensity maps to likelihood of it being part of the lane.

        Each pixel gets its own score, stored as pixel intensity. An intensity of zero means it is not from the lane,
        and a higher score means higher confidence of being from the lane.

        :param img: an image of a road, typically from an overhead perspective.
        :return: The score image.
        """
        # Settings to run thresholding operations on
        # yintian 150 200 180
        # qingtian 160 220 210
        cv2.imwrite("./pic_watch/before.png", img)
        settings = [{'name': 'lab_b', 'cspace': 'LAB', 'channel': 2, 'clipLimit': 2.0, 'threshold': 150},  # 150
                    {'name': 'value', 'cspace': 'HSV', 'channel': 2, 'clipLimit': 6.0, 'threshold': 220},  # 220
                    {'name': 'lightness', 'cspace': 'HLS', 'channel': 1, 'clipLimit': 2.0, 'threshold': 210}]  # 210

        # Perform binary thresholding according to each setting and combine them into one image.
        scores = np.zeros(img.shape[0:2]).astype('uint8')
        for params in settings:
            # Change color space
            color_t = getattr(cv2, 'COLOR_RGB2{}'.format(params['cspace']))
            gray = cv2.cvtColor(img, color_t)[:, :, params['channel']]
            # cv2.imshow(params['cspace'],cv2.cvtColor(img, color_t))
            # Normalize regions of the image using CLAHE
            clahe = cv2.createCLAHE(params['clipLimit'], tileGridSize=(8, 8))
            norm_img = clahe.apply(gray)

            # Threshold to binary
            ret, binary = cv2.threshold(norm_img, params['threshold'], 1, cv2.THRESH_BINARY)
            # cv2.imshow(params['name'],cv2.normalize(binary, None, 0, 255, cv2.NORM_MINMAX))
            scores += binary
        return cv2.normalize(scores, None, 0, 255, cv2.NORM_MINMAX)

    def fit_curve(self, points_x, points_y):
        fit_vals = dict()
        x = np.array(points_x)
        y = np.array(points_y)
        fit_cr = np.polyfit(x, y, 2)
        fit_vals['a'] = fit_cr[0]
        fit_vals['b'] = fit_cr[1]
        fit_vals['x0'] = fit_cr[2]
        return fit_vals

    def fit_lanes(self, points_left, points_right, point_line, fit_globally=False):
        """
        Applies and returns a polynomial fit for given points along the left and right lane line.

        Both lanes are described by a second order polynomial x(y) = ay^2 + by + x0. In the `fit_globally` case,
        a and b are modeled as equal, making the lines perfectly parallel. Otherwise, each line is fit independent of
        the other. The parameters of the model are returned in a dictionary with keys 'al', 'bl', 'x0l' for the left
        lane parameters and 'ar', 'br', 'x0r' for the right lane.

        :param points_left: Two lists of the x and y positions along the left lane line.
        :param points_right: Two lists of the x and y positions along the right lane line.
        :param fit_globally: Set True to use the global, parallel line fit model. In practice this does not allays work.
        :return: fit_vals, a dictionary containing the fitting parameters for the left and right lane as above.
        """

        xl, yl = points_left
        xr, yr = points_right

        for i in range(len(xl)):
            cv2.circle(point_line, (int(xl[i]), yl[i]), 1, (0, 0, 0), 2)
        for i in range(len(xr)):
            cv2.circle(point_line, (int(xr[i]), yr[i]), 1, (0, 0, 0), 2)
        cv2.imwrite('./pic_watch/point_line.png', point_line)

        fit_vals = dict()

        x = np.array(xl)
        y = np.array(yl)
        fit_cr = np.polyfit(y, x, 2)
        fit_vals['al'] = fit_cr[0]
        fit_vals['bl'] = fit_cr[1]
        fit_vals['x0l'] = fit_cr[2]

        x = np.array(xr)
        y = np.array(yr)
        fit_cr = np.polyfit(y, x, 2)
        fit_vals['ar'] = fit_cr[0]
        fit_vals['br'] = fit_cr[1]
        fit_vals['x0r'] = fit_cr[2]

        return fit_vals

    def calc_curvature(self, windows):
        """
        Given a list of Windows along a lane, returns an estimated radius of curvature of the lane.

        Radius of curvature is found by transforming the x,y positions of the windows to the world space, applying
        a simple polynomial fit, and then using the fit values to find curvature.

        :param windows: A List of Windows along a single lane.
        :return: Radius of curvature, in meters.
        """

        x, y = zip(*[window.pos_xy() for window in windows])
        x = np.array(x)
        y = np.array(y)
        fit_cr = np.polyfit(y * self.camera.y_m_per_pix, x * self.camera.x_m_per_pix, 2)
        y_eval = np.max(y)
        return ((1 + (2 * fit_cr[0] * y_eval * self.camera.y_m_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * fit_cr[0])

    def viz_windows(self, score_img, mode):
        """Displays the position of the windows over a score image."""
        if mode == 'filtered':
            lw_img = window_image(self.windows_left, 'x_filtered', color=(0, 255, 0))
            rw_img = window_image(self.windows_right, 'x_filtered', color=(0, 255, 0))
        elif mode == 'raw':
            color = (255, 0, 0)
            win_left_detected, arg = filter_window_list(self.windows_left, False, False, remove_undetected=True)
            win_right_detected, arg = filter_window_list(self.windows_right, False, False, remove_undetected=True)
            lw_img = window_image(win_left_detected, 'x_measured', color, color, color)
            rw_img = window_image(win_right_detected, 'x_measured', color, color, color)
        else:
            raise Exception('mode is not valid')
        combined = lw_img + rw_img
        return cv2.addWeighted(score_img, 1, combined, 0.5, 0)

    def viz_find_lines(self, img):
        """Runs `self.find_lines()` for a single visual, and returns it."""
        y_fit, x_fit_center, x_fit_center_derivative, status = self.find_lines(img)
        return y_fit, x_fit_center, x_fit_center_derivative, status

    def viz_callback(self, visual='presentation'):
        """
        Returns a callback function that takes an image, runs `self.find_lines()` and returns the requested visual.
        """
        return lambda img: self.viz_find_lines(img)


def viz_lane(undist_img, camera, left_fit_x, right_fit_x, fit_y, fit_x_center, mode):
    """
    Take an undistorted dashboard camera image and highlights the lane.

    Code from Udacity SDC-ND Term 1 course code.

    :param undist_img: An undistorted dashboard view image.
    :param camera: The DashboardCamera object for the camera the image was taken on.
    :param left_fit_x: the x values for the left line polynomial at the given y values.
    :param right_fit_x: the x values for the right line polynomial at the given y values.
    :param fit_y: the y values the left and right line x values were calculated at.
    :return: The undistorted image with the lane overlaid on top of it.
    """
    # Create an undist_img to draw the lines on
    lane_poly_overhead = np.zeros_like(undist_img).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([np.array(left_fit_x), fit_y]))])
    pts_right = np.array([np.transpose(np.vstack([right_fit_x, fit_y]))])
    pts_center = np.array([np.transpose(np.vstack([fit_x_center, fit_y]))])
    lane_poly_overhead = cv2.polylines(lane_poly_overhead, np.int32([pts_left]), True, (255, 255, 255), 1)
    lane_poly_overhead = cv2.polylines(lane_poly_overhead, np.int32([pts_right]), True, (255, 255, 255), 1)
    lane_poly_overhead = cv2.polylines(lane_poly_overhead, np.int32([pts_center]), True, (255, 255, 255), 2)

    length = len(pts_center[0])
    center_point = pts_center[0][length // 2]
    cv2.circle(lane_poly_overhead, (int(center_point[0]), int(center_point[1])), 3, (255, 255, 255), 3)
    cv2.imwrite('./pic_watch/test.png', lane_poly_overhead)
    # pts = np.hstack((pts_left, pts_right))
    # cv2.fillPoly(lane_poly_overhead, np.int_([pts]), (255, 0, 255))

    # Warp back to original undist_img space
    if mode == "overhead":
        lane_poly_dash = lane_poly_overhead
    else:
        lane_poly_dash = camera.warp_to_dashboard(lane_poly_overhead)

    # Combine the result with the original undist_img
    return cv2.addWeighted(undist_img, 0.3, lane_poly_dash, 1, 0)

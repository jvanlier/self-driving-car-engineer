from enum import Enum

import numpy as np
import cv2

from .constants import IMG_SHAPE, XM_PER_PIX, YM_PER_PIX

# Sliding window hyperparameters
# Choose the number of sliding windows
NWINDOWS = 9
# Set the width of the windows +/- margin
MARGIN = 120
# Set minimum number of pixels found to recenter window
MINPIX = 50
# Where to evaluate Radius of Curvature:
ROC_Y_EVAL = IMG_SHAPE[1] - 1


class LineType(Enum):
    LEFT = 1
    RIGHT = 2

    @classmethod
    def color(cls, line_type):
        if line_type == cls.LEFT:
            return [255, 0, 0]
        elif line_type == cls.RIGHT:
            return [0, 0, 255]

    @classmethod
    def to_str(cls, line_type):
        if line_type == cls.LEFT:
            return "left"
        elif line_type == cls.RIGHT:
            return "right"


class Line:
    def __init__(self, line_type: LineType, x_pixels, y_pixels,
                 sw_fit_viz: np.ndarray):
        self._params_meter = None
        self._params = None

        self.line_type = line_type
        self.sw_fit_viz = sw_fit_viz

        self.params = self._fit_poly(x_pixels, y_pixels)

        self.plot_line_xs, self.plot_line_ys = self._calc_poly(
            self.params, sw_fit_viz.shape[0])

        self._visualize_poly()

    @classmethod
    def from_sliding_window(cls, binary_warped: np.ndarray,
                            line_type: LineType):
        initial_x = cls._determine_intitial_x_position(binary_warped,
                                                       line_type)

        x_pixels, y_pixels, rectangles = \
            cls._sliding_window_pixel_selection(binary_warped, initial_x)

        sw_fit_viz = (np.dstack([binary_warped] * 3) * 255).astype(np.uint8)
        for rect in rectangles:
            cv2.rectangle(sw_fit_viz, *rect, (0, 255, 0), 2)
        sw_fit_viz[y_pixels, x_pixels] = LineType.color(line_type)

        return cls(line_type, x_pixels, y_pixels, sw_fit_viz)

    @staticmethod
    def _determine_intitial_x_position(binary_warped, line_type: LineType):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[
            binary_warped.shape[0]//2:, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)

        if line_type == LineType.LEFT:
            return np.argmax(histogram[:midpoint])
        elif line_type == LineType.RIGHT:
            return np.argmax(histogram[midpoint:]) + midpoint
        else:
            raise ValueError(f"Unsupported LineType: {line_type}")

    @staticmethod
    def _sliding_window_pixel_selection(binary_warped, initial_x):
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // NWINDOWS)
        # Identify the x and y positions of all nonzero (i.e. activated)
        # pixels in the image:
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        x_current = initial_x
        rectangles = []
        lane_inds = []

        for window in range(NWINDOWS):
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_x_low = x_current - MARGIN
            win_x_high = x_current + MARGIN

            # TODO: draw rectangle
            rectangles.append(((win_x_low, win_y_low),
                               (win_x_high, win_y_high)))

            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_low) & (nonzerox < win_x_high)
                         ).nonzero()[0]

            lane_inds.append(good_inds)

            if len(good_inds) > MINPIX:
                x_current = np.int(np.mean(nonzerox[good_inds]))

        lane_inds = np.concatenate(lane_inds)

        return nonzerox[lane_inds], nonzeroy[lane_inds], rectangles

    @staticmethod
    def _fit_poly(x_pixels, y_pixels):
        return np.polyfit(y_pixels, x_pixels, 2)

    @staticmethod
    def _calc_poly(params, dim_y):
        ys = np.arange(0, dim_y - 1, dtype=np.int32)
        xs = np.round(np.polyval(params, ys)).astype(np.int32)
        return xs, ys

    def _visualize_poly(self):
        line_pts = (np.asarray([self.plot_line_xs, 
                                self.plot_line_ys]).T).astype(np.int32)

        return cv2.polylines(self.sw_fit_viz, [line_pts], False, (255, 255, 0),
                             thickness=5)

    def update_from_prior(self):
        pass

    @property
    def radius_of_curvature(self):
        """Radius of curvature in real-world kilometers."""
        numerator = (
            1 + (2 * self.params_meter[0] +
                 self.params_meter[1]) ** 2) ** 1.5
        denom = np.abs(2 * self.params_meter[0])
        return (numerator / denom) / 1000

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, val):
        self._params = val
        self._params_meter = \
            np.array([val[0] * XM_PER_PIX / YM_PER_PIX ** 2,
                      val[1] * XM_PER_PIX / YM_PER_PIX,
                      val[2]])

    @property
    def params_meter(self):
        return self._params_meter


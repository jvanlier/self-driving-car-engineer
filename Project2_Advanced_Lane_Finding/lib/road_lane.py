import numpy as np
from copy import deepcopy

from .constants import (IMG_SHAPE, XM_PER_PIX,
                        NUM_BAD_FRAMES_UNTIL_RESET,
                        LANE_DIST_MIN_M, LANE_DIST_MAX_M)
from .line import Line, LineType


class RoadLaneMetricNotAvailable(Exception):
    pass


class RoadLane:
    def __init__(self):
        self.left = None
        self.right = None
        self.x_middle = IMG_SHAPE[0] // 2
        self.y_max = IMG_SHAPE[1] - 1
        self.num_bad_frames = 0

    @property
    def radius_of_curvature(self) -> float:
        """Mean of left & right radius of curvature (in kms)."""
        if not self.left or not self.right:
            raise RoadLaneMetricNotAvailable

        return np.mean([self.left.radius_of_curvature,
                        self.right.radius_of_curvature])

    @property
    def vehicle_rel_position(self) -> float:
        """Relative vehicle position, in meters, assuming camera is in
        middle.
        """
        if not self.left or not self.right:
            raise RoadLaneMetricNotAvailable

        y = np.array([self.y_max])
        left_x_bottom = np.polyval(self.left.params, y)[0]
        right_x_bottom = np.polyval(self.right.params, y)[0]
        car_pos_x = (right_x_bottom - left_x_bottom) / 2 + left_x_bottom

        car_deviation_center = car_pos_x - self.x_middle
        # Positive: to the right, negative: to the left

        return car_deviation_center * XM_PER_PIX

    def update(self, binary_warped: np.ndarray) -> str:
        """Update lane lines with new frame.

        Performs sanity checks and resets state if required.

        Args:
            binary_warped: perspective transformed image, binary, with lane
                pixels highlighted.

        Returns:
            status message
        """
        if not self.left or not self.right or \
                self.num_bad_frames >= NUM_BAD_FRAMES_UNTIL_RESET:
            self.left = Line.from_sliding_window(binary_warped,
                                                 LineType.LEFT)
            self.right = Line.from_sliding_window(binary_warped,
                                                  LineType.RIGHT)
            self.num_bad_frames = 0
            return "init from sliding window"

        left_backup = deepcopy(self.left)
        right_backup = deepcopy(self.right)

        self.left.update_from_prior(binary_warped)
        self.right.update_from_prior(binary_warped)

        if self._lane_lines_sane():
            self.num_bad_frames = 0
            return "re-fit with LAF - ok"
        else:
            self.left = left_backup
            self.right = right_backup
            self.num_bad_frames += 1
            return "re-fit with LAF - NOT OK " \
                   f"[{self.num_bad_frames}]"

    def _lane_lines_sane(self) -> bool:
        # Evaluate on top and bottom, check distance in meters
        left_x_top = self.left.plot_line_xs[0]
        left_x_bottom = self.left.plot_line_xs[self.y_max]

        right_x_top = self.right.plot_line_xs[0]
        right_x_bottom = self.right.plot_line_xs[self.y_max]

        distance_top = (right_x_top - left_x_top) * XM_PER_PIX
        distance_bottom = (right_x_bottom - left_x_bottom) * XM_PER_PIX

        if not LANE_DIST_MIN_M < distance_top < LANE_DIST_MAX_M \
                or not LANE_DIST_MIN_M < distance_bottom < LANE_DIST_MAX_M:
            return False
        return True

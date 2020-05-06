import numpy as np

from .constants import IMG_SHAPE, XM_PER_PIX
from .line import Line


class RoadLane:
    def __init__(self, left: Line, right: Line):
        self.left = left
        self.right = right
        self.x_middle = IMG_SHAPE[0] // 2
        self.y_max = IMG_SHAPE[1] - 1

    @property
    def radius_of_curvature(self):
        """Mean of left & right radius of curvature (in kms)."""
        return np.mean([self.left.radius_of_curvature,
                        self.right.radius_of_curvature])

    @property
    def vehicle_rel_position(self):
        """Relative vehicle position, in meters, assuming camera is in
        middle.
        """
        y = np.array([self.y_max])
        left_x_bottom = np.polyval(self.left.params, y)[0]
        right_x_bottom = np.polyval(self.right.params, y)[0]
        car_pos_x = (right_x_bottom - left_x_bottom) / 2 + left_x_bottom

        car_deviation_center = car_pos_x - self.x_middle
        # Positive: to the right, negative: to the left

        return car_deviation_center * XM_PER_PIX

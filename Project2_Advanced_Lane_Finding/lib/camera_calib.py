from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt


GRID_X = 9
GRID_Y = 6


class Undistorter:
    def __init__(self, img_shape: Tuple[int, int]):
        self.img_shape = img_shape
        self.mtx = None
        self.dist = None
        self._calibrated = False

    def calibrate(self, image_paths: List[Path], draw: bool = False):
        objpoints, imgpoints = self._determine_objpoints_imgpoints(
            image_paths, draw)

        ret, self.mtx, self.dist, _, _, = cv2\
            .calibrateCamera(objpoints, imgpoints, self.img_shape, None, None)

        self._calibrated = True

    def apply(self, img):
        if not self._calibrated:
            raise ValueError("Calibrate first!")

        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    @staticmethod
    def _determine_objpoints_imgpoints(image_paths: List[Path],
                                       draw: bool) -> \
            Tuple[np.ndarray, np.ndarray]:
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((GRID_Y * GRID_X, 3), np.float32)
        objp[:, :2] = np.mgrid[0:GRID_X, 0:GRID_Y].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Step through the list and search for chessboard corners
        for fname in image_paths:
            img = cv2.imread(str(fname))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (GRID_X, GRID_Y), None)

            # If found, add object points, image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

                if draw:
                    img = cv2.drawChessboardCorners(img, (GRID_X, GRID_Y), corners, ret)
                    plt.title(fname.name)
                    plt.imshow(img)
                    plt.show()

        return objpoints, imgpoints

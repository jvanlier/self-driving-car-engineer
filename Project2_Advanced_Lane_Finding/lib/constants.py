import numpy as np

IMG_SHAPE = (1280, 720)  # x, y

# Lane pixel extraction thresholds:
SATURATION_THRESHOLD = (155, 255)
SOBEL_X_ABS_SCALED_THRESHOLD = (40, 140)

# Key x and y coordinates for perspective transform:
LANE_START_X_LEFT = 185
LANE_START_X_RIGHT = IMG_SHAPE[0] - 150
LANE_WIDTH = LANE_START_X_RIGHT - LANE_START_X_LEFT
X_MIDDLE = (LANE_START_X_LEFT + LANE_WIDTH // 2)

APEX_Y = 450
APEX_X_OFFSET_LEFT = 65  # Relative to middle of lane
APEX_X_OFFSET_RIGHT = 30

DST_X_OFFSET = 300
DST_X_LEFT = DST_X_OFFSET
DST_X_RIGHT = IMG_SHAPE[0] - DST_X_OFFSET

# Perspective transform points (also usable as polygons):
# Counter clockwise from topleft:
LANE_AREA_SRC = np.array([
    (X_MIDDLE - APEX_X_OFFSET_LEFT, APEX_Y),
    (LANE_START_X_LEFT, IMG_SHAPE[1] - 1),
    (LANE_START_X_RIGHT, IMG_SHAPE[1] - 1),
    (X_MIDDLE + APEX_X_OFFSET_RIGHT, APEX_Y),
], dtype=np.float32)

LANE_AREA_DST = np.array([
    (DST_X_LEFT, 0),
    (DST_X_LEFT, IMG_SHAPE[1] - 1),
    (DST_X_RIGHT, IMG_SHAPE[1] - 1),
    (DST_X_RIGHT, 0),
], dtype=np.float32)

# From warped (perspective transformed) pixel space to approximate
# real-world space:
YM_PER_PIX = 30 / IMG_SHAPE[1]
XM_PER_PIX = 3.7 / (DST_X_RIGHT - DST_X_LEFT)

# Sanity check for distance in meters between lane lines:
# US spec is 3.7 m
LANE_DIST_MIN_M = 2.7
LANE_DIST_MAX_M = 4.7

# Number of "bad" frames we're willing to tolerate in the video before
# resetting:
NUM_BAD_FRAMES_UNTIL_RESET = 25

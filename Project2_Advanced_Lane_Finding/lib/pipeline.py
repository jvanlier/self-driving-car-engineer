import cv2
import numpy as np


def binarize_lane_line_pixels(img, output_binary=False, s_thresh=(170, 255),
                              sx_thresh=(20, 100)) -> np.ndarray:
    """Binarize lane line pixels using Sobel X thresholding and S-channel
    thresholding.

    Args:
        image: the image in BGR (loaded with cv2)
        output_binary: if True, will output a binary image with only 1
            channel, pixel value being max of thresholded sobel and
            thresholded saturation.
            Otherwise, will output image with channel 0 = zeros,
            1 = thresholded Sobel X, 2 = thresholded S (saturation)

            Plotted with matplotlib, Green = Sobel x, Blue = Saturation.
        s_thresh: threshold for S channel
        sx_thresh: threshold for Sobel x

    Return:
        numpy array with a single channel or 3 channels
        (if output_binary=False).
        In either case, values are only 1 or 0.
    """
    img = np.copy(img)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel x on L channel:
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    # Absolute x derivative to accentuate lines away from horizontal:
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= sx_thresh[0]) &
              (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold on S (saturation) channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    if output_binary:
        img_bin = np.zeros((img.shape[0], img.shape[1]))
        img_bin[(sx_binary == 1) | (s_binary == 1)] = 1
        return img_bin
    else:
        # Stack channels such that result of each operation can be inspected
        # individually.
        return np.dstack((np.zeros_like(sx_binary), sx_binary, s_binary)) * 255


class BirdsEyeTansformer:
    def __init__(self, src, dst):
        self._m = cv2.getPerspectiveTransform(src, dst)

    def apply(self, img: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(img, self._m, (img.shape[1], img.shape[0]),
                                   flags=cv2.INTER_LINEAR)

    


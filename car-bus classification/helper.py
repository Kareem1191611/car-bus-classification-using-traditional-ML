from skimage.feature import local_binary_pattern
import cv2
import numpy as np

def compute_lbp_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute LBP features
    lbp_radius = 3
    lbp_points = 8 * lbp_radius
    lbp = local_binary_pattern(gray, lbp_points, lbp_radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, lbp_points + 3), range=(0, lbp_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)
    return lbp_hist




def color_hist(image, nbins=600, bins_range=(0, 256)):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    channel1_hist = np.histogram(image[:,:,0], bins=nbins, range=bins_range)#hue
    channel2_hist = np.histogram(image[:,:,1], bins=nbins, range=bins_range)#saturation
    channel3_hist = np.histogram(image[:,:,2], bins=nbins, range=bins_range)#contrast
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features


def bin_spatial(image, size=(16, 16)):
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    return cv2.resize(image, size).ravel()
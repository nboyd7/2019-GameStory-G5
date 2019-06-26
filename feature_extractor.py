import cv2
import numpy as np
import matplotlib.pyplot as plt

ROI_CROSSHAIR = [75,250,150,500]

# crop frame based on region of interest
def get_subregion_frame(frame, roi=ROI_CROSSHAIR):
    return frame[roi[0]:roi[1],roi[2]:roi[3]]

# compute histogram for any extracted histogram frame (helper function)
def compute_histogram(frame, channels, bins, ranges):
    # We return the histogram as a single vector, in which the three sub-histograms are concatenated.
    histogram = np.zeros(np.sum(bins))

    # We generate a histogram per channel, and then add it to the single-vector histogram.
    for i in range(0, len(channels)):
        channel = channels[i]
        channel_bins = bins[i]
        channel_range = ranges[i]
        channel_histogram = cv2.calcHist(
            [frame],
            [channel],
            None,  # one could specify an optional mask here (we don't use this here),
            [channel_bins],
            channel_range
        )

        # Now we copy these values to the right indices in our single-vector histogram.
        start_index = int(np.sum(bins[0:channel]))
        end_index = start_index + channel_bins
        histogram[start_index:end_index] = channel_histogram.flatten()

    return histogram

# apply histogram type to frame of video then compute
def extract_frame_histogram(frame, channels, bins, ranges, type=cv2.COLOR_BGR2HSV):
    hist_frame = cv2.cvtColor(frame, type)
    histogram = compute_histogram(hist_frame, channels, bins, ranges)
    return histogram / np.sum(histogram)

def extract_greyscale_frame(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def extract_frame_edges(frame):
    edged_frame = cv2.Canny(frame, 50, 200)
    return edged_frame


def histograms_similarity(histogram0, histogram1):
    return cv2.compareHist(histogram0.astype(np.float32), histogram1.astype(np.float32), 2)

def compute_self_similarity(feature_vector_matrix, similarity_function=histograms_similarity):
    similarity_matrix = np.zeros((feature_vector_matrix.shape[0], feature_vector_matrix.shape[0]))
    for i in range(len(feature_vector_matrix)):
        for j in range(len(feature_vector_matrix)):
            similarity_matrix[i, j] = similarity_function(feature_vector_matrix[i], feature_vector_matrix[j])
    # For calling the similarity function, you can just use similarity_function(first_vector, second_vector).
    return similarity_matrix






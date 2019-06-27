import cv2
from scipy.spatial import distance
import numpy as np
import video_handler
import read_metadata
from imutils.video import FileVideoStream
import time

ROI_CROSSHAIR = [75,250,150,500]
COMMENTATOR_STREAM_FP = '/Volumes/Other 1/2018-03-02_P11.mp4'

def video_extract_features(vid_path, frame_number, end_frame):
    # FASTER video reading (decode in separate thread)
    fvs = FileVideoStream(vid_path, start_frame=frame_number).start()
    time.sleep(1.0)

    current_frame = frame_number - 1
    frameCount = end_frame - frame_number + 1

    roi = [0.35, 0.65, 0.35, 0.65]
    frame_width = int(fvs.get_width())
    frame_height = int(fvs.get_height())
    result_width = round(roi[3] * frame_width) - round(roi[2] * frame_width)
    result_height = round(roi[1] * frame_height) - round(roi[0] * frame_height)
    buf = np.empty((frameCount, result_height, result_width), np.dtype('uint8'))
    hist_buf = np.empty((frameCount, 16))
    print_iter = 0

    while fvs.more():

        print_iter += 1
        # Capture frame-by-frame
        frame = fvs.read()
        current_frame += 1

        frame_roi = get_ROI(frame, roi)

        harris_result = get_harris_feature(frame_roi)

        hist_result = extract_frame_histogram(frame_roi)

        buf[current_frame-frame_number] = harris_result
        hist_buf[current_frame - frame_number] = hist_result

        # cv2.imshow('Frame', harris_result)
        if divmod(print_iter, 60)[1] == 0:
            print(f'Progress: {100*(current_frame-frame_number)/(end_frame-frame_number)}%')


        # Press Q on keyboard to  exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print(f'current_frame: {current_frame}')
            break
        if current_frame == end_frame:
            fvs.stop()
            break
    #save numpy matrix of feature frames
    np.save('library_match_1_round_1_harris.npy', buf)
    np.save('library_match_1_round_1_histogram.npy', hist_buf)

# crop frame based on region of interest
def get_subregion_frame(frame, roi=ROI_CROSSHAIR):
    return frame[roi[0]:roi[1], roi[2]:roi[3]]

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

def get_ROI(frame, roi):
    size_1 = frame.shape[0]
    size_2 = frame.shape[1]
    return frame[round(roi[0] * size_1):round(roi[1] * size_1),
            round(roi[2] * size_2):round(roi[3] * size_2)]

def get_harris_feature(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 4, 3, 0.04)
    return np.uint8(255 * dst / dst.max())

# apply histogram type to frame of video then compute
def extract_frame_histogram(frame, channels=[0, 1], bins=[8, 8], ranges=[[0, 180], [0, 256]], type=cv2.COLOR_BGR2HSV):
    hist_frame = cv2.cvtColor(frame, type)
    # cv2.imshow("histogram",hist_frame)
    histogram = compute_histogram(hist_frame, channels, bins, ranges)
    return histogram / np.sum(histogram)

def extract_greyscale_frame(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
# auto edge detector
def extract_frame_edges(frame, sigma=0.33):
    v = np.median(frame)

    # apply automatic Canny edge detectin using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged_frame = cv2.Canny(frame, lower, upper)

    return edged_frame

def histograms_similarity(histogram0, histogram1):
    return cv2.compareHist(histogram0.astype(np.float32), histogram1.astype(np.float32), 2)

# common similarity function to use distance package cosine, manhattan, etc
def similiarity(vector1, vector2, distance_func=distance.cosine):
    return 1 - distance_func(vector1, vector2)

def compute_self_similarity(feature_vector_matrix, similarity_function):
    similarity_matrix = np.zeros((feature_vector_matrix.shape[0], feature_vector_matrix.shape[0]))
    for i in range(len(feature_vector_matrix)):
        for j in range(len(feature_vector_matrix)):
            similarity_matrix[i, j] = similarity_function(feature_vector_matrix[i], feature_vector_matrix[j])
    # For calling the similarity function, you can just use similarity_function(first_vector, second_vector).
    return similarity_matrix

if __name__ == '__main__':
    vid = video_handler.read_video(COMMENTATOR_STREAM_FP)
    ss = read_metadata.StreamSync()
    fn = ss.GetSyncMatchFilename(1, 11)
    video_handler.display_video_extract(vid, 374565, 376565, extract_frame_edges)









import numpy as np
import read_JSON
import utils as u
import video_handler
import cv2
import glob
import feature_extractor as ft

def getDiffError(frame, query_frame):
    return np.sum(np.abs(np.subtract(frame, query_frame)))

feature_config = {
    'harris': {
        'error_fn': getDiffError,
        'match_threshold': 100000,
        'library': './library_match_1_round_1_harris.npy',
        'type': 'distance'
    },
    'histogram': {
        'error_fn': ft.histograms_similarity,
        'match_threshold': 0.9,
        'library': './library_match_1_round_1_histogram.npy',
        'type': 'similarity'
    }
}

def findMatch(library, query, ft_name):
    differences = []
    for frame in library:
        differences.append(feature_config[ft_name]['error_fn'](frame, query))
    np_diff = np.array(differences)
    if feature_config[ft_name]['type'] == 'distance':
        the_frame = np.argmin(np_diff)
    elif feature_config[ft_name]['type'] == 'similarity':
        the_frame = np.argmax(np_diff)
    return the_frame, differences[the_frame], differences

def frameOffsetToCommentatorFrame(match_id, frame_offset):
    match_start = u.commentator_stream_match_bounds[f'match_{match_id}'][0]
    return match_start + frame_offset

def findEventsInLibrary():
    dH = read_JSON.DataHandler()
    libraries = {}
    print('Loading libraries...')
    for ft_name in feature_config:
        libraries[ft_name] = np.load(feature_config[ft_name]['library'])
        print(libraries[ft_name].shape)
    print(' loaded _<')

    for actor in dH.eventWindowArrayPerActor:
        window_event_features = ['dummy']
        win_indx = 1
        while len(window_event_features) > 0:
            for ft_name in feature_config.keys():
                window_event_features = glob.glob(f'./resources/event_queries/{actor}_w{win_indx}_e*_{ft_name}.npy')
                if len(window_event_features) == 0:
                    break
                matched, scores, frames = findWindowInLibrary(libraries[ft_name], window_event_features, ft_name)
                print(f'file: {window_event_features} ft_name: {ft_name} match: {matched} scores: {scores} ')
                for frame in frames:
                    print(f'frame: {frameOffsetToCommentatorFrame(1,frame)}')
            win_indx += 1
            break
        break

def findWindowInLibrary(library, window_event_features, ft_name):
    match_scores = []
    matched_frames = []
    for window_event_feature in window_event_features:
        # check if feature in library
        query = np.load(window_event_feature)
        closest_match_frame_offset, closest_match_diff, all_match_scores = findMatch(library, query, ft_name)
        match_scores.append(closest_match_diff)
        matched_frames.append(closest_match_frame_offset)
    return [np.asarray(match_scores)< feature_config[ft_name]['match_threshold']], match_scores, matched_frames


def showMatchedFrame(vid_path, closest_match_frame_offset):
    commentator_frame = frameOffsetToCommentatorFrame(1, closest_match_frame_offset)
    video_handler.read_frame(vid_path, commentator_frame)

def createEventFeatures():
    dH = read_JSON.DataHandler()
    ds = dH.eventWindowArrayPerActor

    roi = [0.35, 0.65, 0.35, 0.65]

    for actor in ds:
        vid_path = u.playerToPath(actor)
        window_count = 0
        for window in ds[actor]:
            window_count += 1
            event_count = 0
            for event_time in window['event_times']:
                event_count += 1
                event_frame = u.event_time_to_stream_frame(event_time, u.player_id[actor], 1)
                frame = video_handler.get_frame(vid_path, event_frame)

                frame_roi = ft.get_ROI(frame, roi)

                harris_result = ft.get_harris_feature(frame_roi)

                hist_result = ft.extract_frame_histogram(frame_roi)

                np.save(f'./resources/event_queries/{actor}_w{window_count}_e{event_count}_harris', harris_result)
                np.save(f'./resources/event_queries/{actor}_w{window_count}_e{event_count}_histogram', hist_result)


if __name__ == '__main__':
    # library = np.load('./library_match_1_round_1_harris.npy')
    findEventsInLibrary()

    # video_handler.read_frame('C:\\save\\2018-03-02_P11.mp4', 237176)
    # video_handler.read_frame('C:\\save\\2018-03-02_P11.mp4', 237288)
    # video_handler.read_frame('C:\\save\\2018-03-02_P11.mp4', 237239)
    # video_handler.read_frame('C:\\save\\2018-03-02_P11.mp4', 237226)

    # findEventsInLibrary(library)
    # library = np.load('features_1.npy')
    # query = np.load('resources/event_queries/olofmeister_w1_e1.npy')
    #
    # res = []
    # iter = 0
    # for frame in library:
    #     res.append(np.sum(np.abs(np.subtract(frame, query))))
    #     if res[iter] == 164265:
    #         print(iter)
    #     iter += 1
    #
    # print(sorted(res))
    # print(min(res))
    # vid_path = 'C:\save\\2018-03-02_P11.mp4'

    # video_handler.read_frame(vid_path, 236500, frame_name="Frame")

    # video_handler.read_frame(vid_path, 374500, frame_name="Frame")
    # video_handler.display_video(vid_path, 375347)
    # video_handler.display_video_extract(vid_path, 237062 + 60, 237062 + 300)

    # dH = read_JSON.DataHandler()
    # vid_path = 'C:\\save\\2018-03-02_P11.mp4'
    # ft.video_extract_features(vid_path, u.commentator_stream_match_bounds['match_1'][0], u.commentator_stream_match_bounds['match_1'][0]+2400)

    # ds = dH.eventWindowArrayPerActor
    #
    # for actor in ds:
    #     vid_path = u.playerToPath(actor)
    #     window_count = 0
    #     for window in ds[actor]:
    #         window_count += 1
    #         event_count = 0
    #         for event_time in window['event_times']:
    #             event_count += 1
    #             event_frame = u.event_time_to_stream_frame(event_time, u.player_id[actor], 1)
    #             frame = video_handler.get_frame(vid_path, event_frame)
    #
    #             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #             gray = np.float32(gray)
    #             roi = [0.35, 0.65, 0.35,
    #                    0.65]
    #             size_1 = gray.shape[0]
    #             size_2 = gray.shape[1]
    #
    #             gray = gray[round(roi[0] * size_1):round(roi[1] * size_1),
    #                    round(roi[2] * size_2):round(roi[3] * size_2)]
    #             dst = cv2.cornerHarris(gray, 4, 3, 0.04)
    #             harris_result = np.uint8(255 * dst / dst.max())
    #             np.save(f'./resources/event_queries/{actor}_w{window_count}_e{event_count}', harris_result)




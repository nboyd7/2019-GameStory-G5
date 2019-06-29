import json
import numpy as np
import read_JSON
import utils as u
import video_handler
import cv2
import glob
import feature_extractor as ft
from PIL import Image
import matplotlib.pyplot as plt
import skimage.measure
import collections
import operator




overall_match_threshold = 0.75 * len(u.feature_config.keys())

com_path = 'C:\save\\2018-03-02_P11.mp4'



def findWindowInLibrary(libraries, window_event_features):
    match_scores = []
    matched_frames = []
    for window_event_feature in window_event_features:
        # check if feature in library
        queries = {}
        for ft_name in u.feature_config.keys():
            queries[ft_name] = np.load(f'{window_event_feature[:-10]}{ft_name}.npy')
        closest_match_frame_offset, closest_match_diff = u.findMatch(libraries, queries)
        match_scores.append(np.sum(closest_match_diff))
        matched_frames.append(closest_match_frame_offset)
    return [np.asarray(match_scores) < overall_match_threshold], match_scores, matched_frames

def findEventsInLibrary():
    dH = read_JSON.DataHandler()
    libraries = {}
    print('Loading libraries...')
    for ft_name in u.feature_config:
        print(ft_name)
        libraries[ft_name] = np.load(u.feature_config[ft_name]['library'])
        print(libraries[ft_name].shape)
    print(' loaded _<')

    for actor in dH.eventWindowArrayPerActor:
        window_event_features = ['dummy']
        win_indx = 1
        while len(window_event_features) > 0:
            window_event_features = glob.glob(f'./resources/event_queries/{actor}_w{win_indx}_e*_harris.npy')
            if len(window_event_features) == 0:
                break
            matched, scores, frames = findWindowInLibrary(libraries, window_event_features)
            print(f'file: {window_event_features} match: {matched} scores: {scores} ')
            for frame in frames:
                print(f'frame: {frameOffsetToCommentatorFrame(1,frame)}')
                video_handler.read_frame(com_path, u.frameOffsetToCommentatorFrame(1,frame), f'{window_event_features[0]} events: {len(window_event_features)}')
            win_indx += 1


def showMatchedFrame(vid_path, closest_match_frame_offset):
    commentator_frame = u.frameOffsetToCommentatorFrame(1, closest_match_frame_offset)
    video_handler.read_frame(vid_path, commentator_frame)

def createEventFeatures():
    eventData = {}

    dH = read_JSON.DataHandler()
    ds = dH.eventWindowArrayPerActor

    roi = [0.25, 0.55, 0.35, 0.65]

    for actor in ds:
        eventData[actor] = {}
        vid_path = u.playerToPath(actor)
        window_count = 0
        for window in ds[actor]:
            window_count += 1
            eventData[actor][f'w_{window_count}'] = {}
            event_count = 0
            for event_time in window['event_times']:
                event_count += 1

                event_frame = u.event_time_to_stream_frame(event_time, u.player_id[actor], 1)
                frame = video_handler.get_frame(vid_path, event_frame)

                frame_roi = ft.get_ROI(frame, roi)

                harris_result = ft.get_harris_feature(frame_roi)

                hist_result = ft.extract_frame_histogram(frame_roi)

                # canny_result = ft.get_canny_feature(frame_roi)

                harris_filename = f'./resources/event_queries/{actor}_w{window_count}_e{event_count}_harris'
                hist_filename = f'./resources/event_queries/{actor}_w{window_count}_e{event_count}_histogram'

                eventData[actor][f'w_{window_count}'][f'e_{event_count}'] = {
                    'event_frame': event_frame,
                    'harris_filename': harris_filename,
                    'hist_filename': hist_filename
                }

                np.save(harris_filename, harris_result)
                np.save(hist_filename, hist_result)

                # np.save(f'./resources/event_queries/{actor}_w{window_count}_e{event_count}_canny', canny_result)
    #dump event data json
    with open('stored_json\\player_event_features.txt', 'w') as outfile:
        json.dump(eventData, outfile)

# match 1 approx 144000 frames
def buildLibrary(start_frame=u.commentator_stream_match_bounds['match_1'][0], end_frame=u.commentator_stream_match_bounds['match_1'][0]+144000):
    ft.video_extract_features(com_path, start_frame, end_frame)

def saveDuo(frame1, frame2, filename='match.png', titles=['Player_stream', 'Commentator_stream']):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.title.set_text(titles[0])
    ax2.title.set_text(titles[1])

    plt.subplot(121)
    plt.axis('off')
    plt.imshow(frame1)
    plt.subplot(122)
    plt.axis('off')
    plt.imshow(frame2)

    plt.savefig(f'output\\{filename}')

def saveTrio(frame1, frame2, frame3, filename='ft_match.png', titles=['Player_stream', 'Commentator_stream', 'Difference'], subfolder=None):
    fig = plt.figure()

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.title.set_text(titles[0])
    ax2.title.set_text(titles[1])
    ax3.title.set_text(titles[2])


    plt.subplot(131)
    plt.axis('off')
    plt.imshow(frame1)
    plt.subplot(132)
    plt.axis('off')
    plt.imshow(frame2)
    plt.subplot(133)
    plt.axis('off')
    plt.imshow(frame3)

    if subfolder != None:
        filename = subfolder + '\\' + filename
    plt.savefig(f'output\\{filename}')
    plt.close(fig)

def saveNineGrid(images=[], filename='nine_grid', titles=None, subfolder=None):
    fig = plt.figure()

    ax1 = fig.add_subplot(331)
    ax2 = fig.add_subplot(332)
    ax3 = fig.add_subplot(333)

    ax4 = fig.add_subplot(334)
    ax5 = fig.add_subplot(335)
    ax6 = fig.add_subplot(336)

    ax7 = fig.add_subplot(337)
    ax8 = fig.add_subplot(338)
    ax9 = fig.add_subplot(339)

    if titles != None:
        ax1.title.set_text(titles[0])
        ax2.title.set_text(titles[1])
        ax3.title.set_text(titles[2])
        ax4.title.set_text(titles[3])
        ax5.title.set_text(titles[4])
        ax6.title.set_text(titles[5])
        ax7.title.set_text(titles[6])
        ax8.title.set_text(titles[7])
        ax9.title.set_text(titles[8])

    plt.subplot(331)
    plt.axis('off')
    plt.imshow(images[0])
    plt.subplot(332)
    plt.axis('off')
    plt.imshow(images[1])
    plt.subplot(333)
    plt.axis('off')
    plt.imshow(images[2])

    plt.subplot(334)
    plt.axis('off')
    plt.imshow(images[3])
    plt.subplot(335)
    plt.axis('off')
    plt.imshow(images[4])
    plt.subplot(336)
    plt.axis('off')
    plt.imshow(images[5])

    plt.subplot(337)
    plt.axis('off')
    plt.imshow(images[6])
    plt.subplot(338)
    plt.axis('off')
    plt.imshow(images[7])
    plt.subplot(339)
    plt.axis('off')
    plt.imshow(images[8])

    if subfolder != None:
        filename = subfolder + '\\' + filename
    plt.savefig(f'output\\{filename}')
    plt.close(fig)

def saveNineGridHist(data=[], filename='nine_grid', titles=None, subfolder=None):
    fig = plt.figure()

    ax1 = fig.add_subplot(331)
    ax2 = fig.add_subplot(332)
    ax3 = fig.add_subplot(333)

    ax4 = fig.add_subplot(334)
    ax5 = fig.add_subplot(335)
    ax6 = fig.add_subplot(336)

    ax7 = fig.add_subplot(337)
    ax8 = fig.add_subplot(338)
    ax9 = fig.add_subplot(339)

    if titles != None:
        ax1.title.set_text(titles[0])
        ax2.title.set_text(titles[1])
        ax3.title.set_text(titles[2])
        ax4.title.set_text(titles[3])
        ax5.title.set_text(titles[4])
        ax6.title.set_text(titles[5])
        ax7.title.set_text(titles[6])
        ax8.title.set_text(titles[7])
        ax9.title.set_text(titles[8])

    plt.subplot(331)
    plt.axis('off')
    plt.hist(data[0])
    plt.subplot(332)
    plt.axis('off')
    plt.hist(data[1])
    plt.subplot(333)
    plt.axis('off')
    plt.hist(data[2])

    plt.subplot(334)
    plt.axis('off')
    plt.hist(data[3])
    plt.subplot(335)
    plt.axis('off')
    plt.hist(data[4])
    plt.subplot(336)
    plt.axis('off')
    plt.hist(data[5])

    plt.subplot(337)
    plt.axis('off')
    plt.hist(data[6])
    plt.subplot(338)
    plt.axis('off')
    plt.hist(data[7])
    plt.subplot(339)
    plt.axis('off')
    plt.hist(data[8])

    if subfolder != None:
        filename = subfolder + '\\' + filename
    plt.savefig(f'output\\{filename}')
    plt.close(fig)


def edgeCountDiff(edges_1, edges_2):
    hor_diff_norm = np.abs(edges_1[0] - edges_2[0])/max(edges_1[0], edges_2[0])
    vert_diff_norm = np.abs(edges_1[1] - edges_2[1])/max(edges_1[1], edges_2[1])
    ratio_1 = edges_1[0]/edges_1[1]
    ratio_2 = edges_2[0]/edges_2[1]
    ratio_diff_norm = np.abs(ratio_1-ratio_2)/max(ratio_1, ratio_2)
    return ratio_diff_norm

if __name__ == '__main__':
    print("main method")
    # buildLibrary()
    # createEventFeatures()
    u.storeWindowMatches()
    # library = np.load('./library_match_1_round_1_harris.npy')
    # findEventsInLibrary()

    # libraries = {}
    # queries = {}
    # libraries['harris'] = np.load('./library_match_1_round_1_harris.npy')
    # queries['harris'] = np.load('./resources/event_queries\\olofmeister_w1_e1_harris.npy')
    # a, b, diffs = findMatch(libraries, queries)
    #
    # saveTrio(queries['harris'], libraries['harris'][a], np.abs(np.subtract(libraries['harris'][a], queries['harris'])))





    #  FEATURE PARAM SELECTION manual GRID SEARCH
    # com_frame_guard_w2_e1 = 240987
    # com_frame_olof_w1_e1 = 237176
    # com_frame_rain_w1_e1 = frameOffsetToCommentatorFrame(1, 451342 - 443324)
    #
    # guard_frame = 447854
    # olof_frame = 444023
    # rain_frame = 451342
    #
    # olof_path = 'C:\save\\2018-03-02_P7.mp4'
    # guard_path = 'C:\save\\2018-03-02_P6.mp4'
    # rain_path = 'C:\save\\2018-03-02_P10.mp4'
    #
    # frame1_1 = video_handler.get_frame(olof_path, olof_frame)
    # frame1_2 = video_handler.get_frame(com_path, com_frame_olof_w1_e1)
    # frame2_1 = video_handler.get_frame(guard_path, guard_frame)
    # frame2_2 = video_handler.get_frame(com_path, com_frame_guard_w2_e1)
    # frame3_1 = video_handler.get_frame(rain_path, rain_frame)
    # frame3_2 = video_handler.get_frame(com_path, com_frame_rain_w1_e1)

    # edge_count_1_1 = ft.count_canny_edges_per_direction(ft.get_canny_feature(frame1_1, threshold1=50, threshold2=100, blurIn=5, blurOut=15))
    # edge_count_1_2 = ft.count_canny_edges_per_direction(ft.get_canny_feature(frame1_2, threshold1=50, threshold2=100, blurIn=5, blurOut=15))
    # edge_count_2_1 = ft.count_canny_edges_per_direction(ft.get_canny_feature(frame2_1, threshold1=50, threshold2=100, blurIn=5, blurOut=15))
    # edge_count_2_2 = ft.count_canny_edges_per_direction(ft.get_canny_feature(frame2_2, threshold1=50, threshold2=100, blurIn=5, blurOut=15))
    # edge_count_3_1 = ft.count_canny_edges_per_direction(ft.get_canny_feature(frame3_1, threshold1=50, threshold2=100, blurIn=5, blurOut=15))
    # edge_count_3_2 = ft.count_canny_edges_per_direction(ft.get_canny_feature(frame3_2, threshold1=50, threshold2=100, blurIn=5, blurOut=15))
    #
    # ed_1_1 = edgeCountDiff(edge_count_1_1,  edge_count_1_2)
    # ed_1_2 = edgeCountDiff(edge_count_1_1,  edge_count_2_2)
    # ed_1_3 = edgeCountDiff(edge_count_1_1,  edge_count_3_2)
    # ed_2_1 = edgeCountDiff(edge_count_2_1,  edge_count_1_2)
    # ed_2_2 = edgeCountDiff(edge_count_2_1,  edge_count_2_2)
    # ed_2_3 = edgeCountDiff(edge_count_2_1,  edge_count_3_2)
    # ed_3_1 = edgeCountDiff(edge_count_3_1,  edge_count_1_2)
    # ed_3_2 = edgeCountDiff(edge_count_3_1,  edge_count_2_2)
    # ed_3_3 = edgeCountDiff(edge_count_3_1,  edge_count_3_2)

    # print(f'{ed_1_1} {ed_1_2} {ed_1_3}')
    # print(f'{ed_2_1} {ed_2_2} {ed_2_3}')
    # print(f'{ed_3_1} {ed_3_2} {ed_3_3}')

    # #get harris of frames with different params and save trio diff
    # maxVarOnDiff = 0
    # iter = 0
    # # channels = [0, 1], bins = [8, 8], ranges = [[0, 180], [0, 256]], type = cv2.COLOR_BGR2HSV
    # p_ch = [[0, 1], [0, 2], [1, 2], [0], [1], [2], [0, 1, 2]]
    # p_binSize = [4, 6, 8, 10, 12]
    # p_types = [cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2HLS]
    # ranges = [[0, 180], [0, 256], [0, 256]]
    #
    # s_combo = {}
    #
    # for channels in p_ch:
    #     for binSize in p_binSize:
    #         for c_type in p_types:
    #             bins = [binSize] * len(channels)
    #
    #             # print(f'channels{channels}_binSize{binSize}_c_type{c_type}_bins{bins}')
    #             # for blur in p_b:
    #             iter = iter + 1
    #             print(f'progress [{100*iter/(len(p_ch)*len(p_binSize)*len(p_types)):.2f}]')
    #             roi = [0.25, 0.55, 0.35, 0.65] #above weapon because different reflection and different weapon flash (on different frames...)
    #
    #             frame_roi = ft.get_ROI(frame1_1, roi)
    #             hist_result_1_1 = ft.extract_frame_histogram(frame_roi, channels=channels, bins=bins, ranges=ranges, type=c_type)
    #             frame_roi = ft.get_ROI(frame1_2, roi)
    #             hist_result_1_2 = ft.extract_frame_histogram(frame_roi, channels=channels, bins=bins, ranges=ranges, type=c_type)
    #             frame_roi = ft.get_ROI(frame2_1, roi)
    #             hist_result_2_1 = ft.extract_frame_histogram(frame_roi, channels=channels, bins=bins, ranges=ranges, type=c_type)
    #             frame_roi = ft.get_ROI(frame2_2, roi)
    #             hist_result_2_2 = ft.extract_frame_histogram(frame_roi, channels=channels, bins=bins, ranges=ranges, type=c_type)
    #             frame_roi = ft.get_ROI(frame3_1, roi)
    #             hist_result_3_1 = ft.extract_frame_histogram(frame_roi, channels=channels, bins=bins, ranges=ranges, type=c_type)
    #             frame_roi = ft.get_ROI(frame3_2, roi)
    #             hist_result_3_2 = ft.extract_frame_histogram(frame_roi, channels=channels, bins=bins, ranges=ranges, type=c_type)
    #
    #             the_diff_1_1 = np.uint8(np.abs(np.subtract(np.int16(hist_result_1_1), np.int16(hist_result_1_2))))
    #             the_diff_1_2 = np.uint8(np.abs(np.subtract(np.int16(hist_result_1_1), np.int16(hist_result_2_2))))
    #
    #             the_diff_1_1 = ft.histograms_similarity(hist_result_1_1, hist_result_1_2)
    #             the_diff_1_2 = ft.histograms_similarity(hist_result_1_1, hist_result_2_2)
    #             the_diff_1_3 = ft.histograms_similarity(hist_result_1_1, hist_result_3_2)
    #             the_diff_2_1 = ft.histograms_similarity(hist_result_2_1, hist_result_1_2)
    #             the_diff_2_2 = ft.histograms_similarity(hist_result_2_1, hist_result_2_2)
    #             the_diff_2_3 = ft.histograms_similarity(hist_result_2_1, hist_result_3_2)
    #             the_diff_3_1 = ft.histograms_similarity(hist_result_3_1, hist_result_1_2)
    #             the_diff_3_2 = ft.histograms_similarity(hist_result_3_1, hist_result_2_2)
    #             the_diff_3_3 = ft.histograms_similarity(hist_result_3_1, hist_result_3_2)
    #
    #             s_m1 = 2*the_diff_1_1 / (the_diff_1_2 + the_diff_1_3)
    #             s_m2 = 2*the_diff_2_2 / (the_diff_2_1 + the_diff_2_3)
    #             s_m3 = 2*the_diff_3_3 / (the_diff_3_1 + the_diff_3_2)
    #
    #             file_name = f'hist_ch{channels}_b{bins}_type{c_type}.png'
    #             s_combo[f'{file_name}'] = (s_m1 + s_m2 + s_m3)/3
    #
    #             # imgarr = [the_diff_1_1, the_diff_1_2, the_diff_1_3, the_diff_2_1, the_diff_2_2, the_diff_2_3, the_diff_3_1, the_diff_3_2, the_diff_3_3,]
    #             # titlearr = ['diff_match', 'diff_noMatch', 'diff_noMatch', 'diff_noMatch', 'diff_match', 'diff_noMatch', 'diff_noMatch', 'diff_noMatch', 'diff_match']
    #
    #             # saveNineGridHist(filename=file_name, data=imgarr, titles=titlearr, subfolder='nine_hist_matchNoMatch')
    # sorted_s_combo = sorted(s_combo.items(), key=operator.itemgetter(1))

    # p_ch = [[0, 1], [0, 2], [1, 2], [0], [1], [2], [0, 1, 2]]
    # p_binSize = [4, 6, 8, 10, 12]
    # p_types = [cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2HLS]
    # ranges = [[0, 180], [0, 256], [0, 256]]
    #
    # s_combo = {}
    #
    # for channels in p_ch:
    #     for binSize in p_binSize:
    #         for c_type in p_types:
    #             bins = [binSize] * len(channels)
    #
    #             # print(f'channels{channels}_binSize{binSize}_c_type{c_type}_bins{bins}')
    #             # for blur in p_b:
    #             iter = iter + 1
    #             print(f'progress [{100*iter/(len(p_ch)*len(p_binSize)*len(p_types)):.2f}]')
    #             roi = [0.25, 0.55, 0.35,
    #                    0.65]  # above weapon because different reflection and different weapon flash (on different frames...)
    #
    #             frame_roi = ft.get_ROI(frame1_1, roi)
    #             hist_result_1_1 = ft.extract_frame_histogram(frame_roi, channels=channels, bins=bins, ranges=ranges,
    #                                                          type=c_type)
    #             frame_roi = ft.get_ROI(frame1_2, roi)
    #             hist_result_1_2 = ft.extract_frame_histogram(frame_roi, channels=channels, bins=bins, ranges=ranges,
    #                                                          type=c_type)
    #             frame_roi = ft.get_ROI(frame2_1, roi)
    #             hist_result_2_1 = ft.extract_frame_histogram(frame_roi, channels=channels, bins=bins, ranges=ranges,
    #                                                          type=c_type)
    #             frame_roi = ft.get_ROI(frame2_2, roi)
    #             hist_result_2_2 = ft.extract_frame_histogram(frame_roi, channels=channels, bins=bins, ranges=ranges,
    #                                                          type=c_type)
    #             frame_roi = ft.get_ROI(frame3_1, roi)
    #             hist_result_3_1 = ft.extract_frame_histogram(frame_roi, channels=channels, bins=bins, ranges=ranges,
    #                                                          type=c_type)
    #             frame_roi = ft.get_ROI(frame3_2, roi)
    #             hist_result_3_2 = ft.extract_frame_histogram(frame_roi, channels=channels, bins=bins, ranges=ranges,
    #                                                          type=c_type)
    #
    #             the_diff_1_1 = np.uint8(np.abs(np.subtract(np.int16(hist_result_1_1), np.int16(hist_result_1_2))))
    #             the_diff_1_2 = np.uint8(np.abs(np.subtract(np.int16(hist_result_1_1), np.int16(hist_result_2_2))))
    #
    #             the_diff_1_1 = ft.histograms_similarity(hist_result_1_1, hist_result_1_2)
    #             the_diff_1_2 = ft.histograms_similarity(hist_result_1_1, hist_result_2_2)
    #             the_diff_1_3 = ft.histograms_similarity(hist_result_1_1, hist_result_3_2)
    #             the_diff_2_1 = ft.histograms_similarity(hist_result_2_1, hist_result_1_2)
    #             the_diff_2_2 = ft.histograms_similarity(hist_result_2_1, hist_result_2_2)
    #             the_diff_2_3 = ft.histograms_similarity(hist_result_2_1, hist_result_3_2)
    #             the_diff_3_1 = ft.histograms_similarity(hist_result_3_1, hist_result_1_2)
    #             the_diff_3_2 = ft.histograms_similarity(hist_result_3_1, hist_result_2_2)
    #             the_diff_3_3 = ft.histograms_similarity(hist_result_3_1, hist_result_3_2)
    #
    #             s_m1 = 2 * the_diff_1_1 / (the_diff_1_2 + the_diff_1_3)
    #             s_m2 = 2 * the_diff_2_2 / (the_diff_2_1 + the_diff_2_3)
    #             s_m3 = 2 * the_diff_3_3 / (the_diff_3_1 + the_diff_3_2)
    #
    #             file_name = f'hist_ch{channels}_b{bins}_type{c_type}.png'
    #             s_combo[f'{file_name}'] = (s_m1 + s_m2 + s_m3) / 3



    # p_th1 = [50, 75, 100, 125, 150]
    # p_th2_delta = [25, 50, 75, 100, 125]
    # p_as = [2, 3, 4, 5, 7, 10, 15, 20]
    # p_blur = [3, 5, 9, 15, 20]
    # p_blur2 = [3, 5, 9, 15, 20]
    # s_combo = {}
    # a_s = 3
    # for th1 in p_th1:
    #     for th2_delta in p_th2_delta:
    #         # for blur in [1]:
    #         for blur2 in p_blur2:
    #             th2 = min(th1 + th2_delta, 255)
    #             # for blur in p_b:
    #             iter = iter + 1
    #             print(f'progress [{100*iter/(len(p_th1)*len(p_th2_delta)*len(p_blur2)):.2f}]')
    #             roi = [0.25, 0.55, 0.35,
    #                    0.65]  # above weapon because different reflection and different weapon flash (on different frames...)
    #
    #             frame_roi = ft.get_ROI(frame1_1, roi)
    #             canny_result_1_1 = cv2.boxFilter(
    #                 ft.get_canny_feature(frame_roi, threshold1=th1, threshold2=th2, apertureSize=a_s), ddepth=-1,
    #                 ksize=(blur2, blur2))
    #             frame_roi = ft.get_ROI(frame1_2, roi)
    #             canny_result_1_2 = cv2.boxFilter(
    #                 ft.get_canny_feature(frame_roi, threshold1=th1, threshold2=th2, apertureSize=a_s), ddepth=-1,
    #                 ksize=(blur2, blur2))
    #             frame_roi = ft.get_ROI(frame2_1, roi)
    #             canny_result_2_1 = cv2.boxFilter(
    #                 ft.get_canny_feature(frame_roi, threshold1=th1, threshold2=th2, apertureSize=a_s), ddepth=-1,
    #                 ksize=(blur2, blur2))
    #             frame_roi = ft.get_ROI(frame2_2, roi)
    #             canny_result_2_2 = cv2.boxFilter(
    #                 ft.get_canny_feature(frame_roi, threshold1=th1, threshold2=th2, apertureSize=a_s), ddepth=-1,
    #                 ksize=(blur2, blur2))
    #             frame_roi = ft.get_ROI(frame3_1, roi)
    #             canny_result_3_1 = cv2.boxFilter(
    #                 ft.get_canny_feature(frame_roi, threshold1=th1, threshold2=th2, apertureSize=a_s), ddepth=-1,
    #                 ksize=(blur2, blur2))
    #             frame_roi = ft.get_ROI(frame3_2, roi)
    #             canny_result_3_2 = cv2.boxFilter(
    #                 ft.get_canny_feature(frame_roi, threshold1=th1, threshold2=th2, apertureSize=a_s), ddepth=-1,
    #                 ksize=(blur2, blur2))
    #
    #             the_diff_1_1 = np.uint8(np.abs(np.subtract(np.int16(canny_result_1_1), np.int16(canny_result_1_2))))
    #             the_diff_1_2 = np.uint8(np.abs(np.subtract(np.int16(canny_result_1_1), np.int16(canny_result_2_2))))
    #             the_diff_1_3 = np.uint8(np.abs(np.subtract(np.int16(canny_result_1_1), np.int16(canny_result_3_2))))
    #
    #             the_diff_2_1 = np.uint8(np.abs(np.subtract(np.int16(canny_result_2_1), np.int16(canny_result_1_2))))
    #             the_diff_2_2 = np.uint8(np.abs(np.subtract(np.int16(canny_result_2_1), np.int16(canny_result_2_2))))
    #             the_diff_2_3 = np.uint8(np.abs(np.subtract(np.int16(canny_result_2_1), np.int16(canny_result_3_2))))
    #
    #             the_diff_3_1 = np.uint8(np.abs(np.subtract(np.int16(canny_result_3_1), np.int16(canny_result_1_2))))
    #             the_diff_3_2 = np.uint8(np.abs(np.subtract(np.int16(canny_result_3_1), np.int16(canny_result_2_2))))
    #             the_diff_3_3 = np.uint8(np.abs(np.subtract(np.int16(canny_result_3_1), np.int16(canny_result_3_2))))
    #
    #             s_m1 = 2 * np.sum(the_diff_1_1) / (
    #                         (np.sum(the_diff_1_2) + np.sum(the_diff_1_3)) * the_diff_1_1.shape[0] * the_diff_1_1.shape[
    #                     1])
    #             s_m2 = 2 * np.sum(the_diff_2_2) / (
    #                         (np.sum(the_diff_2_1) + np.sum(the_diff_2_3)) * the_diff_1_1.shape[0] * the_diff_1_1.shape[
    #                     1])
    #             s_m3 = 2 * np.sum(the_diff_3_3) / (
    #                         (np.sum(the_diff_3_1) + np.sum(the_diff_3_2)) * the_diff_1_1.shape[0] * the_diff_1_1.shape[
    #                     1])
    #
    #             file_name = f'canny_th{th1}-{th2}_b{blur2}.png'
    #             s_combo[f'{file_name}'] = (s_m1 + s_m2 + s_m3) / 3
    #
    #             imgarr = [the_diff_1_1, the_diff_1_2, the_diff_1_3, the_diff_2_1, the_diff_2_2, the_diff_2_3,
    #                       the_diff_3_1, the_diff_3_2, the_diff_3_3, ]
    #             titlearr = ['diff_match', 'diff_noMatch', 'diff_noMatch', 'diff_noMatch', 'diff_match', 'diff_noMatch',
    #                         'diff_noMatch', 'diff_noMatch', 'diff_match']
    #
    #             saveNineGrid(filename=file_name, images=imgarr, titles=titlearr,
    #                          subfolder='nine_canny_matchNoMatch_blur_canny')


    # p_bs = [2, 4, 5, 8, 12, 16]
    # p_as = [1, 3, 5, 15]
    # p_k = [0.03, 0.04, 0.05]
    # p_b = [3, 5, 9, 15, 25]
    # p_pl = [2, 4, 8]
    # s_combo = {}
    # for bs in p_bs:
    #     for a_s in p_as:
    #         for k in p_k:
    #             for blur in p_b:
    # frame_roi = cv2.boxFilter(ft.get_ROI(frame1_1, roi), ddepth=-1, ksize=(blur, blur))
    # harris_result_1_1 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    # frame_roi = cv2.boxFilter(ft.get_ROI(frame1_2, roi), ddepth=-1, ksize=(blur, blur))
    # harris_result_1_2 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    # frame_roi = cv2.boxFilter(ft.get_ROI(frame2_1, roi), ddepth=-1, ksize=(blur, blur))
    # harris_result_2_1 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    # frame_roi = cv2.boxFilter(ft.get_ROI(frame2_2, roi), ddepth=-1, ksize=(blur, blur))
    # harris_result_2_2 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    # frame_roi = cv2.boxFilter(ft.get_ROI(frame3_1, roi), ddepth=-1, ksize=(blur, blur))
    # harris_result_3_1 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    # frame_roi = cv2.boxFilter(ft.get_ROI(frame3_2, roi), ddepth=-1, ksize=(blur, blur))
    # harris_result_3_2 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    #
    # the_diff_1_1 = np.uint8(np.abs(np.subtract(harris_result_1_1, harris_result_1_2)))
    # the_diff_1_2 = np.uint8(np.abs(np.subtract(harris_result_1_1, harris_result_2_2)))
    # the_diff_1_3 = np.uint8(np.abs(np.subtract(harris_result_1_1, harris_result_3_2)))
    #
    # the_diff_2_1 = np.uint8(np.abs(np.subtract(harris_result_2_1, harris_result_1_2)))
    # the_diff_2_2 = np.uint8(np.abs(np.subtract(harris_result_2_1, harris_result_2_2)))
    # the_diff_2_3 = np.uint8(np.abs(np.subtract(harris_result_2_1, harris_result_3_2)))
    #
    # the_diff_3_1 = np.uint8(np.abs(np.subtract(harris_result_3_1, harris_result_1_2)))
    # the_diff_3_2 = np.uint8(np.abs(np.subtract(harris_result_3_1, harris_result_2_2)))
    # the_diff_3_3 = np.uint8(np.abs(np.subtract(harris_result_3_1, harris_result_3_2)))
    #
    # s_m1 = 2 * np.sum(the_diff_1_1) / (
    #             (np.sum(the_diff_1_2) + np.sum(the_diff_1_3)) * the_diff_1_1.shape[0] * the_diff_1_1.shape[1])
    # s_m2 = 2 * np.sum(the_diff_2_2) / (
    #             (np.sum(the_diff_2_1) + np.sum(the_diff_2_3)) * the_diff_1_1.shape[0] * the_diff_1_1.shape[1])
    # s_m3 = 2 * np.sum(the_diff_3_3) / (
    #             (np.sum(the_diff_3_1) + np.sum(the_diff_3_2)) * the_diff_1_1.shape[0] * the_diff_1_1.shape[1])
    #
    # file_name = f'harris_bs{bs}_as{a_s}_k{k}_blur{blur}.png'
    # s_combo[f'{file_name}'] = (s_m1 + s_m2 + s_m3) / 3
    #
    # imgarr = [the_diff_1_1, the_diff_1_2, the_diff_1_3, the_diff_2_1, the_diff_2_2, the_diff_2_3, the_diff_3_1,
    #           the_diff_3_2, the_diff_3_3, ]
    # titlearr = ['diff_match', 'diff_noMatch', 'diff_noMatch', 'diff_noMatch', 'diff_match', 'diff_noMatch',
    #             'diff_noMatch', 'diff_noMatch', 'diff_match']
    #
    # saveNineGrid(filename=file_name, images=imgarr, titles=titlearr,
    #              subfolder='nine_harris_matchNoMatch_blur_harris_input')





    # variance = np.var(harris_result_1)
    # diffRatio = np.sum(the_diff)/((the_diff.shape[0]*the_diff.shape[1]))
    # varONdiffRatio = int(variance/diffRatio)
    # maxVarOnDiff = max(maxVarOnDiff, varONdiffRatio)
    # saveTrio(np.uint8(harris_result_1), np.uint8(harris_result_2), the_diff, filename=f'olof_match_w1_e1_diff_harris_bs{bs}_as{a_s}_k{k}_varONdiffRatio{varONdiffRatio}.png',
    #          titles=['P_stream', 'C_stream', 'Difference'], subfolder='harris_params')





    # frame_roi = ft.get_ROI(frame1_1, roi)
    # harris_result_1_1 = skimage.measure.block_reduce(
    #     np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k)), (pl, pl), np.mean)
    # frame_roi = ft.get_ROI(frame1_2, roi)
    # harris_result_1_2 = skimage.measure.block_reduce(
    #     np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k)), (pl, pl), np.mean)
    # the_diff_1_3 = np.uint8(np.abs(np.subtract(harris_result_1_1, harris_result_1_2)))
    #
    # frame_roi = ft.get_ROI(frame2_1, roi)
    # harris_result_2_1 = skimage.measure.block_reduce(
    #     np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k)), (pl, pl), np.mean)
    # frame_roi = ft.get_ROI(frame2_2, roi)
    # harris_result_2_2 = skimage.measure.block_reduce(
    #     np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k)), (pl, pl), np.mean)
    # the_diff_2_3 = np.uint8(np.abs(np.subtract(harris_result_2_1, harris_result_2_2)))
    #
    # frame_roi = ft.get_ROI(frame3_1, roi)
    # harris_result_3_1 = skimage.measure.block_reduce(
    #     np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k)), (pl, pl), np.mean)
    # frame_roi = ft.get_ROI(frame3_2, roi)
    # harris_result_3_2 = skimage.measure.block_reduce(
    #     np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k)), (pl, pl), np.mean)
    # the_diff_3_3 = np.uint8(np.abs(np.subtract(harris_result_3_1, harris_result_3_2)))
    #
    # file_name = f'harris_bs{bs}_as{a_s}_k{k}_pl{pl}.png'
    # saveNineGrid(filename=file_name, images=[np.uint8(harris_result_1_1), np.uint8(harris_result_1_2), the_diff_1_3,
    #                                          np.uint8(harris_result_2_1), np.uint8(harris_result_2_2), the_diff_2_3,
    #                                          np.uint8(harris_result_3_1), np.uint8(harris_result_3_2), the_diff_3_3, ],
    #              titles=['P frame 1', 'C frame 1', 'Difference', 'P frame 2', 'C frame 2', 'Difference', 'P frame 3',
    #                      'C frame 3', 'Difference'], subfolder='nine_harris_meanpool_harris')









    # saveDuo(frame1, frame2, filename='olof_w1_e1_match.png')
    # saveTrio(frame1, frame2, np.abs(np.subtract(frame1, frame2)), filename='olof_match_w1_e1_diff.png', titles=['P_stream', 'C_stream', 'Difference'])

    # frame_roi = cv2.boxFilter(ft.get_ROI(frame1_1, roi), ddepth=-1, ksize=(blur, blur))
    # harris_result_1_1 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    # frame_roi = cv2.boxFilter(ft.get_ROI(frame1_2, roi), ddepth=-1, ksize=(blur, blur))
    # harris_result_1_2 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    # the_diff_1_3 = np.uint8(np.abs(np.subtract(harris_result_1_1, harris_result_1_2)))
    #
    # frame_roi = cv2.boxFilter(ft.get_ROI(frame2_1, roi), ddepth=-1, ksize=(blur, blur))
    # harris_result_2_1 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    # frame_roi = cv2.boxFilter(ft.get_ROI(frame2_2, roi), ddepth=-1, ksize=(blur, blur))
    # harris_result_2_2 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    # the_diff_2_3 = np.uint8(np.abs(np.subtract(harris_result_2_1, harris_result_2_2)))
    #
    # frame_roi = cv2.boxFilter(ft.get_ROI(frame3_1, roi), ddepth=-1, ksize=(blur, blur))
    # harris_result_3_1 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    # frame_roi = cv2.boxFilter(ft.get_ROI(frame3_2, roi), ddepth=-1, ksize=(blur, blur))
    # harris_result_3_2 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    # the_diff_3_3 = np.uint8(np.abs(np.subtract(harris_result_3_1, harris_result_3_2)))

    # frame_roi = ft.get_ROI(frame1_1, roi)
    # frame_roi = skimage.measure.block_reduce(frame_roi, (pl, pl, 1), np.max)
    # harris_result_1_1 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    # frame_roi = ft.get_ROI(frame1_2, roi)
    # frame_roi = skimage.measure.block_reduce(frame_roi, (pl, pl, 1), np.max)
    # harris_result_1_2 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    # the_diff_1_3 = np.uint8(np.abs(np.subtract(harris_result_1_1, harris_result_1_2)))
    #
    # frame_roi = ft.get_ROI(frame2_1, roi)
    # frame_roi = skimage.measure.block_reduce(frame_roi, (pl, pl, 1), np.max)
    # harris_result_2_1 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    # frame_roi = ft.get_ROI(frame2_2, roi)
    # frame_roi = skimage.measure.block_reduce(frame_roi, (pl, pl, 1), np.max)
    # harris_result_2_2 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    # the_diff_2_3 = np.uint8(np.abs(np.subtract(harris_result_2_1, harris_result_2_2)))
    #
    # frame_roi = ft.get_ROI(frame3_1, roi)
    # frame_roi = skimage.measure.block_reduce(frame_roi, (pl, pl, 1), np.max)
    # harris_result_3_1 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    # frame_roi = ft.get_ROI(frame3_2, roi)
    # frame_roi = skimage.measure.block_reduce(frame_roi, (pl, pl, 1), np.max)
    # harris_result_3_2 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    # the_diff_3_3 = np.uint8(np.abs(np.subtract(harris_result_3_1, harris_result_3_2)))
    #
    # file_name = f'harris_bs{bs}_as{a_s}_k{k}_pl{pl}.png'

    # plt.subplot(311)
    # plt.imshow(queries['harris'])
    # plt.subplot(312)
    # plt.imshow(libraries['harris'][a])
    # plt.subplot(313)
    # plt.imshow(np.abs(np.subtract(libraries['harris'][a], queries['harris'])))
    # plt.show()

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




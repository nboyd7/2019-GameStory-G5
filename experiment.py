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

def frameOffsetToCommentatorFrame(match_id, frame_offset):
    match_start = u.commentator_stream_match_bounds[f'match_{match_id}'][0]
    return match_start + frame_offset

def getDiffError(frame, query_frame):
    return 1 - np.sum(np.abs(np.subtract(frame, query_frame))/255)/(frame.shape[0]*frame.shape[1])


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

com_path = 'C:\save\\2018-03-02_P11.mp4'

def findMatch(libraries, queries):
    differences = []
    for indx in range(0, libraries[next(iter(libraries))].shape[0]):
        frame_scores = []
        for ft_name in libraries.keys():
            frame = libraries[ft_name][indx]
            frame_scores.append(feature_config[ft_name]['error_fn'](frame, queries[ft_name]))
        differences.append(sum(frame_scores)/len(frame_scores))
    np_diff = np.array(differences)
    the_frame = np.argmax(np_diff)
    return the_frame, differences[the_frame], differences

def findWindowInLibrary(libraries, window_event_features):
    match_scores = []
    matched_frames = []
    for window_event_feature in window_event_features:
        # check if feature in library
        queries = {}
        for ft_name in feature_config.keys():
            queries[ft_name] = np.load(f'{window_event_feature[:-10]}{ft_name}.npy')
        closest_match_frame_offset, closest_match_diff, all_match_scores = findMatch(libraries, queries)
        match_scores.append(closest_match_diff)
        matched_frames.append(closest_match_frame_offset)
    return [np.asarray(match_scores) < feature_config[ft_name]['match_threshold']], match_scores, matched_frames

def findEventsInLibrary():
    dH = read_JSON.DataHandler()
    libraries = {}
    print('Loading libraries...')
    for ft_name in feature_config:
        print(ft_name)
        libraries[ft_name] = np.load(feature_config[ft_name]['library'])
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
                video_handler.read_frame(com_path, frameOffsetToCommentatorFrame(1,frame), f'{window_event_features[0]} events: {len(window_event_features)}')
            win_indx += 1




def showMatchedFrame(vid_path, closest_match_frame_offset):
    commentator_frame = frameOffsetToCommentatorFrame(1, closest_match_frame_offset)
    video_handler.read_frame(vid_path, commentator_frame)

def createEventFeatures():
    dH = read_JSON.DataHandler()
    ds = dH.eventWindowArrayPerActor

    roi = [0.25, 0.55, 0.35, 0.65]

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

def buildLibrary(start_frame=u.commentator_stream_match_bounds['match_1'][0], end_frame=u.commentator_stream_match_bounds['match_1'][0]+18000):
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

if __name__ == '__main__':
    # buildLibrary()
    # createEventFeatures()
    # library = np.load('./library_match_1_round_1_harris.npy')
    findEventsInLibrary()

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
    #q
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
    #
    # #get harris of frames with different params and save trio diff
    # maxVarOnDiff = 0
    # iter = 0
    # p_bs = [2, 4, 5, 8, 12, 16]
    # p_as = [1, 3, 5 , 15]
    # p_k = [0.03, 0.04, 0.05]
    # p_b = [3, 5, 9, 15, 25]
    # p_pl = [2, 4, 8]
    # s_combo = {}
    # for bs in p_bs:
    #     for a_s in p_as:
    #         for k in p_k:
    #             for blur in p_b:
    #                 # for blur in p_b:
    #                 iter = iter + 1
    #                 print(f'progress [{100*iter/(len(p_bs)*len(p_as)*len(p_k)*len(p_b)):.2f}]')
    #                 roi = [0.25, 0.55, 0.35, 0.65] #above weapon because different reflection and different weapon flash (on different frames...)
    #
    #                 frame_roi = cv2.boxFilter(ft.get_ROI(frame1_1, roi), ddepth=-1, ksize=(blur, blur))
    #                 harris_result_1_1 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    #                 frame_roi = cv2.boxFilter(ft.get_ROI(frame1_2, roi), ddepth=-1, ksize=(blur, blur))
    #                 harris_result_1_2 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    #                 frame_roi = cv2.boxFilter(ft.get_ROI(frame2_1, roi), ddepth=-1, ksize=(blur, blur))
    #                 harris_result_2_1 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    #                 frame_roi = cv2.boxFilter(ft.get_ROI(frame2_2, roi), ddepth=-1, ksize=(blur, blur))
    #                 harris_result_2_2 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    #                 frame_roi = cv2.boxFilter(ft.get_ROI(frame3_1, roi), ddepth=-1, ksize=(blur, blur))
    #                 harris_result_3_1 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    #                 frame_roi = cv2.boxFilter(ft.get_ROI(frame3_2, roi), ddepth=-1, ksize=(blur, blur))
    #                 harris_result_3_2 = np.int16(ft.get_harris_feature(frame_roi, blockSize=bs, aperture_size=a_s, k=k))
    #
    #                 the_diff_1_1 = np.uint8(np.abs(np.subtract(harris_result_1_1, harris_result_1_2)))
    #                 the_diff_1_2 = np.uint8(np.abs(np.subtract(harris_result_1_1, harris_result_2_2)))
    #                 the_diff_1_3 = np.uint8(np.abs(np.subtract(harris_result_1_1, harris_result_3_2)))
    #
    #                 the_diff_2_1 = np.uint8(np.abs(np.subtract(harris_result_2_1, harris_result_1_2)))
    #                 the_diff_2_2 = np.uint8(np.abs(np.subtract(harris_result_2_1, harris_result_2_2)))
    #                 the_diff_2_3 = np.uint8(np.abs(np.subtract(harris_result_2_1, harris_result_3_2)))
    #
    #                 the_diff_3_1 = np.uint8(np.abs(np.subtract(harris_result_3_1, harris_result_1_2)))
    #                 the_diff_3_2 = np.uint8(np.abs(np.subtract(harris_result_3_1, harris_result_2_2)))
    #                 the_diff_3_3 = np.uint8(np.abs(np.subtract(harris_result_3_1, harris_result_3_2)))
    #
    #                 s_m1 = 2 * np.sum(the_diff_1_1) / ((np.sum(the_diff_1_2) + np.sum(the_diff_1_3)) * the_diff_1_1.shape[0] * the_diff_1_1.shape[1])
    #                 s_m2 = 2 * np.sum(the_diff_2_2) / ((np.sum(the_diff_2_1) + np.sum(the_diff_2_3)) * the_diff_1_1.shape[0] * the_diff_1_1.shape[1])
    #                 s_m3 = 2 * np.sum(the_diff_3_3) / ((np.sum(the_diff_3_1) + np.sum(the_diff_3_2)) * the_diff_1_1.shape[0] * the_diff_1_1.shape[1])
    #
    #                 file_name = f'harris_bs{bs}_as{a_s}_k{k}_blur{blur}.png'
    #                 s_combo[f'{file_name}'] = (s_m1 + s_m2 + s_m3)/3
    #
    #                 imgarr = [the_diff_1_1, the_diff_1_2, the_diff_1_3, the_diff_2_1, the_diff_2_2, the_diff_2_3, the_diff_3_1, the_diff_3_2, the_diff_3_3,]
    #                 titlearr = ['diff_match', 'diff_noMatch', 'diff_noMatch', 'diff_noMatch', 'diff_match', 'diff_noMatch', 'diff_noMatch', 'diff_noMatch', 'diff_match']
    #
    #                 saveNineGrid(filename=file_name, images=imgarr, titles=titlearr, subfolder='nine_harris_matchNoMatch_blur_harris_input')
    # sorted_s_combo = sorted(s_combo.items(), key=operator.itemgetter(1))

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




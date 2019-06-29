import os
import json
from datetime import datetime
import numpy as np
import feature_extractor as ft
import video_handler as vidH
import utils as u
import read_JSON

#master start frame is the start frame to which the json event log time is synced
master_start_frame = [443259]


#format: stream_starts[video_name][match_id] = match_round_1_start_frame
stream_starts = {
    '2018-03-02_P1': [443285],
    '2018-03-02_P2': [443323],
    '2018-03-02_P3': [443296],
    '2018-03-02_P4': [443292],
    '2018-03-02_P5': [443258],
    '2018-03-02_P6': [443240],
    '2018-03-02_P7': [443235],
    '2018-03-02_P8': [443220],
    '2018-03-02_P9': [443235],
    '2018-03-02_P10': [443216],
    '2018-03-02_P11': [438875]
}

commentator_stream_match_bounds = {
    'match_1': [236500, 383096]
}
#format: event_log_match_start_time[match_id] = event_log_time corresponding to match_round_1_start_frame
event_log_match_start_time = [datetime(2018, 3, 2, 12, 9, 55, 350000)]

player_id = {
    'RUSH'       : 1,
    'tarik'      : 2,
    'Skadoodle'  : 3,
    'Stewie2k'   : 4,
    'autimatic'  : 5,
    'GuardiaN'   : 6,
    'olofmeister': 7,
    'NiKo'       : 8,
    'karrigan'   : 9,
     'rain'      : 10
}

def getDiffError(frame, query_frame):
    return 1 - np.sum(np.abs(np.subtract(frame, query_frame))/255)/(frame.shape[0]*frame.shape[1])


feature_config = {
    'harris': {
        'error_fn': getDiffError,
        'match_threshold': 0.8,
        'library': './library_match_1_segment_harris.npy',
    },
    'histogram': {
        'error_fn': ft.histograms_similarity,
        'match_threshold': 0.8,
        'library': './library_match_1_segment_histogram.npy',
    },
    # 'canny': {
    #     'error_fn': getDiffError,
    #     'match_threshold': 0.8,
    #     'library': './library_match_1_segment_canny.npy',
    # }
}


def storeWindowMatches():
    myDh = read_JSON.DataHandler()
    libraries = {'harris': np.load('./library_match_1_segment_harris.npy'),
                 'histogram': np.load('./library_match_1_segment_histogram.npy')}
    windowMatches = {}
    ds = myDh.eventWindowArrayPerActor
    total = 0
    for actor in ds:
        for data in ds[actor]:
            total+=1

    iter = 0
    for actor in ds:
        windowMatches[actor] = {}
        for data in ds[actor]:
            found, frames, scores = findWindow(libraries, data, actor)
            win_id = data['window']
            windowMatches[actor][f'w_{win_id}'] = {
                'matched': found,
                'frames': list(map(int, frames)),
                'scores': list(map(float, scores))
            }
            for e in range(0, len(frames)):
                windowMatches[actor][f'w_{win_id}'][f'e_{e+1}'] = {}
                windowMatches[actor][f'w_{win_id}'][f'e_{e+1}']['frame'] = int(frames[e])
                windowMatches[actor][f'w_{win_id}'][f'e_{e+1}']['score'] = float(scores[e])
            iter += 1
            print(f'progress {iter/total:.2f}')
    with open('stored_json\\player_window_event_matches.txt', 'w') as outfile:
        json.dump(windowMatches, outfile)


def updateWindowMatches(newThresh):
    with open('stored_json\\player_window_event_matches.txt') as json_file:
        data = json.load(json_file)
    for actor in data:
        for window in data[actor]:
            data[actor][window]['matched'] = isaMatch(data[actor][window]['scores'], data[actor][window]['frames'], newThresh)
    with open('stored_json\\player_window_event_matches.txt', 'w') as outfile:
        json.dump(data, outfile)
    print('Window matches updated with new threshold')

def findWindow(libraries, data, player_name):
    stored_window_match = u.getWindowStoredData(player_name, data['window'])
    if stored_window_match != None:
        return stored_window_match['matched'], stored_window_match['frames'], stored_window_match['scores']
    print("window match not found in json, finding window...")
    match_thresh = 0.8
    roi = [0.25, 0.55, 0.35, 0.65]
    vid_path = u.playerToPath(player_name)
    i = 0
    closest_match_frame_offset = []
    closest_match_diff = []
    for event_time in data['event_times']:
        i += 1
        stored_event = u.getEventStoredData(player_name, data['window'], i)
        if stored_event != None:
            harris_result = np.load(stored_event['harris_filename'] + '.npy')
            hist_result = np.load(stored_event['hist_filename'] + '.npy')
            event_frame = stored_event['event_frame']
            print('loaded event')
        else:
            event_frame = u.event_time_to_stream_frame(event_time, u.player_id[player_name], 1)
            frame = vidH.get_frame(vid_path, event_frame)
            frame_roi = ft.get_ROI(frame, roi)
            harris_result = ft.get_harris_feature(frame_roi)
            hist_result = ft.extract_frame_histogram(frame_roi)
            print('computed event features')

        queries = {'harris': harris_result, 'histogram': hist_result}
        cmo, cmd = findNewMatch(libraries, queries)
        closest_match_frame_offset.append(cmo)
        closest_match_diff.append(cmd)
    return isaMatch(closest_match_diff, closest_match_frame_offset, match_thresh), closest_match_frame_offset, closest_match_diff

def isaMatch(closest_match_diff, closest_match_frame_offset, match_thresh):
    if sum(closest_match_diff) / len(closest_match_diff) > match_thresh:
        if len(closest_match_frame_offset) > 1:
            for i in range(0, len(closest_match_frame_offset) - 1):
                frame_diff = closest_match_frame_offset[i + 1] - closest_match_frame_offset[i]
                if frame_diff < 0:
                    return False
    return True


def frameOffsetToCommentatorFrame(match_id, frame_offset):
    match_start = u.commentator_stream_match_bounds[f'match_{match_id}'][0]
    return match_start + frame_offset


def getMatch(libraries, data, player_name, event_number):
    roi = [0.25, 0.55, 0.35, 0.65]
    vid_path = u.playerToPath(player_name)
    stored_event = u.getEventStoredData(player_name, data['window'], event_number)
    if stored_event != None:
        print('loaded event')
        harris_result = np.load(stored_event['harris_filename'] + '.npy')
        hist_result = np.load(stored_event['hist_filename'] + '.npy')
        event_frame = stored_event['event_frame']
    else:
        event_frame = u.event_time_to_stream_frame(data['event_time'], u.player_id[player_name], 1)
        frame = vidH.get_frame(vid_path, event_frame)
        frame_roi = ft.get_ROI(frame, roi)
        harris_result = ft.get_harris_feature(frame_roi)
        hist_result = ft.extract_frame_histogram(frame_roi)
        print('computed event features')
    queries = {'harris': harris_result, 'histogram': hist_result}
    closest_match_frame_offset, closest_match_diff = findMatch(libraries, queries, player_name, data, event_number)
    vidH.read_frame('C:\save\\2018-03-02_P11.mp4', frameOffsetToCommentatorFrame(1, closest_match_frame_offset),
                             f'{player_name} event {event_number} frame match score {closest_match_diff}')

def findMatch(libraries, queries, player_name, data, event_number):
    stored_window_match = u.getWindowStoredData(player_name, data['window'])
    if stored_window_match != None:
        if f'e_{event_number}' in stored_window_match:
            return stored_window_match[f'e_{event_number}']['frame'], stored_window_match[f'e_{event_number}']['score']
    differences = []
    for indx in range(0, libraries[next(iter(libraries))].shape[0]):
        frame_scores = []
        for ft_name in libraries.keys():
            frame = libraries[ft_name][indx]
            frame_scores.append(feature_config[ft_name]['error_fn'](frame, queries[ft_name]))
        differences.append(sum(frame_scores)/len(frame_scores))
    np_diff = np.array(differences)
    the_frame = np.argmax(np_diff)
    return the_frame, differences[the_frame]

def findNewMatch(libraries, queries):
    differences = []
    for indx in range(0, libraries[next(iter(libraries))].shape[0]):
        frame_scores = []
        for ft_name in libraries.keys():
            frame = libraries[ft_name][indx]
            frame_scores.append(feature_config[ft_name]['error_fn'](frame, queries[ft_name]))
        differences.append(sum(frame_scores)/len(frame_scores))
    np_diff = np.array(differences)
    the_frame = np.argmax(np_diff)
    return the_frame, differences[the_frame]

def getEventStoredData(actor='', window=1, event=1):
    file_path = 'stored_json\\player_event_features.txt'
    if os.path.exists(file_path):
        with open(file_path) as json_file:
            data = json.load(json_file)
            if actor in data:
                if f'w_{window}' in data[actor]:
                    if f'e_{event}' in data[actor][f'w_{window}']:
                        return data[actor][f'w_{window}'][f'e_{event}']
    print(f'Data not found for params: actor_{actor} window_{window} event_{event}')
    return None

def getWindowStoredData(actor='', window=1):
    file_path = 'stored_json\\player_window_event_matches.txt'
    if os.path.exists(file_path):
        with open(file_path) as json_file:
            data = json.load(json_file)
            if actor in data:
                if f'w_{window}' in data[actor]:
                    return data[actor][f'w_{window}']
    print(f'Data not found for params: actor_{actor} window_{window}')
    return None

def getRankedList():
    myDh = read_JSON.DataHandler()
    ds = myDh.eventWindowArrayPerActor
    stats = {}
    for actor in ds:
        for data in ds[actor]:
            stored_window_match = u.getWindowStoredData(actor, data['window'])
            if stored_window_match != None:
                if stored_window_match['matched']:
                    window_stats = {}
                    window_stats['weight'] = data['total_weight']
                    window_stats['data'] = data
                    window_stats['actor'] = actor
                    stats[f'{actor}_{window}'] = window_stats
    sorted_keys = sorted(stats, key=lambda x: (stats[x]['weight']))
    result_dict_list = []
    for key in sorted_keys:
        temp_dict = stats[key]
        temp_dict['window_name'] = key
        result_dict_list.append(temp_dict)
    return temp_dict

def playerToPath(player_name):
    id = player_id[player_name]
    path = f'C:\save\\2018-03-02_P{id}.mp4'
    return path

def convertDate(indate):
    return datetime.strptime(indate[:-7] + '0000', '%Y-%m-%dT%H:%M:%S.%f')

def event_time_to_stream_frame(dt, player_id, match_id):
    match_indx = match_id-1
    vid_stream_path = f'C:\save\\2018-03-02_P{player_id}.mp4'
    time_from_match_start = dt - event_log_match_start_time[match_indx]
    frame_from_match_start = round(time_from_match_start.seconds * 60 + (time_from_match_start.microseconds / 1000000) * 60)
    player_stream_frame_offset_from_master = stream_starts[f'2018-03-02_P{player_id}'][match_indx] - master_start_frame[match_indx]
    frame_from_player_stream_match_start = stream_starts[f'2018-03-02_P{player_id}'][match_indx] + player_stream_frame_offset_from_master + frame_from_match_start
    return frame_from_player_stream_match_start
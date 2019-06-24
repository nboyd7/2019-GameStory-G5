# Given event find it in the video files
from datetime import timedelta


import tkinter as tk
import cv2
import utils
from datetime import datetime
import read_JSON

START_TIME_DAY_1 = '2018-03-02T10:00:00.000+00:00'
START_TIME_DAY_2 = '2018-03-04T10:00:00.000+00:00'


# todo: make a dictionary that matches player names with the video file they are in P1 - P10
#format is stream_player[player_id][match_indx] = player_name
stream_player = {
    '2018-03-02_P1': ['RUSH'],
    '2018-03-02_P2': ['tarik'],
    '2018-03-02_P3': ['Skadoodle'],
    '2018-03-02_P4': ['Stweie2k'],
    '2018-03-02_P5': ['autimatic'],
    '2018-03-02_P6': ['GuardiaN'],
    '2018-03-02_P7': ['olofmeister'],
    '2018-03-02_P8': ['NiKo'],
    '2018-03-02_P9': ['karrigan'],
    '2018-03-02_P10': ['rain']
}

player_stream = []
player_stream.append( {
    'RUSH'       : '2018-03-02_P1',
    'tarik'      : '2018-03-02_P2',
    'Skadoodle'  : '2018-03-02_P3',
    'Stewie2k'   : '2018-03-02_P4',
    'autimatic'  : '2018-03-02_P5',
    'GuardiaN'   : '2018-03-02_P6',
    'olofmeister': '2018-03-02_P7',
    'NiKo'       : '2018-03-02_P8',
    'karrigan'   : '2018-03-02_P9',
    'rain'      : '2018-03-02_P10'
} )

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



video_finder_day_1 = {
    'RUSH':'2018-03-02_P1.mp4',
    'nitr0':'2018-03-02_P1.mp4',
    'tarik':'2018-03-02_P2.mp4',
    'ELiGE':'2018-03-02_P2.mp4',
    'Skadoodle':'2018-03-02_P3.mp4',
    'twistzz':'2018-03-02_P3.mp4',
    'Stewie2K':'2018-03-02_P4.mp4',
    'steal':'2018-03-02_P4.mp4'
}

video_finder_day_2 = {
    'RUSH': '2018-03-02_P1.mp4',
    'nitr0': '2018-03-02_P1.mp4',
    'tarik': '2018-03-02_P2.mp4',
    'ELiGE': '2018-03-02_P2.mp4',
    'Skadoodle': '2018-03-02_P3.mp4',
    'twistzz': '2018-03-02_P3.mp4',
    'Stewie2K': '2018-03-02_P4.mp4',
    'steal': '2018-03-02_P4.mp4'
}


# return cv2 video capture object
def read_video(fp):
    vid = cv2.VideoCapture(fp)
    if (vid.isOpened() == False):
        print("Error opening video stream or file")
    return vid

# https://www.geeksforgeeks.org/python-play-a-video-using-opencv/
# diplay video for a certain frame number
def display_video(cap, frame_number, title='Frame'):
    # Read until video is completed
    cap.set(1, frame_number - 1)
    current_frame = frame_number - 1
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        current_frame += 1

        if ret == True:
            # Display the resulting frame
            cv2.imshow(title, frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print(f'current_frame: {current_frame}')
                break
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def display_video_extract(vid_path, frame_number, end_frame, title='Frame'):
    cap = read_video(vid_path)
    # Read until video is completed
    cap.set(1, frame_number - 1)
    current_frame = frame_number - 1
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        current_frame += 1

        if ret == True:
            # Display the resulting frame
            cv2.imshow(title, frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print(f'current_frame: {current_frame}')
                break
        # Break the loop
        else:
            break

        if current_frame == end_frame:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

#given event object find the starting frame.
# Use the vid capture object to calculate fps.
# Using start date of the streams
# (hard coded - 10:00 for both stream days)
def find_frame(event, start_date, vid):
    date_str = event['date']
    date = utils.convertDate(date_str)
    delta = get_time_delta(start_date, date)
    return calculate_frames(delta, vid.get(cv2.CAP_PROP_FPS))

#using time delta calulate the frames based off of fps of the video
def calculate_frames(delta, fps):
    seconds = delta.total_seconds()
    return fps * seconds

def get_time_delta(start_date, cur_date):
    return cur_date - start_date


# get video file based on actor of event
def get_video_file(actor, day=1):
    if day == 1:
        return video_finder_day_1[actor]
    else:
        return video_finder_day_2[actor]

#get start of round frame based of round number
def get_start_match(match_num, frame_dict):
    return frame_dict[match_num]

def get_frame_offset(json_event_datatime, json_start_datetime, fps=60):
    return timedelta(json_event_datatime - json_start_datetime) * fps


# return frame of even
def sync(event, frame_dict, vid, start_date):
    start_round_frame = get_start_match(event["round"], frame_dict)


def read_frame(cap, frame_number, frame_name="Frame"):
    cap.set(1, frame_number)
    ret, frame = cap.read()

    cv2.imshow(frame_name, frame)

    cv2.waitKey()

    cap.release()
    cv2.destroyAllWindows()


def displayEventsDatasetDemo(display_events=10, show_frames_before=120):
    myDh = read_JSON.DataHandler()

    ds = myDh.dataset
    display_event_count = 0
    for event in ds:
        event_time = event['date']
        event_playerid = player_id[event['actor']]
        event_filename = f"2018-03-02_P{event_playerid}"
        event_framenumber = utils.event_time_to_stream_frame(event_time, event_playerid, 1)

        video_path = f'C:\save\\{event_filename}.mp4'
        vid = read_video(video_path)

        event_type = event['event_type']
        event_actor = event['actor']
        display_video(vid, event_framenumber - show_frames_before, title=f'{event_type} {event_actor}')

        display_event_count += 1
        if display_event_count == display_events:
            break

if __name__ == '__main__':
    myDl = read_JSON.DataHandler()
    myEventWindows = myDl.eventWindowArrayPerActor

    """for actor in myDl.actors:
        print(actor)
        for i in range(0, len(myDl.eventWindowArrayPerActor[actor])):
            if myDl.eventWindowArrayPerActor[actor][i]["total_weight"] > 0:
                print(myDl.eventWindowArrayPerActor[actor][i])"""












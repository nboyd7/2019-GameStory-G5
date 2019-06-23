# Given event find it in the video files
from datetime import timedelta

import cv2
import utils

START_TIME_DAY_1 = '2018-03-02T10:00:00.000+00:00'
START_TIME_DAY_2 = '2018-03-04T10:00:00.000+00:00'


# todo: make a dictionary that matches player names with the video file they are in P1 - P10
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
def display_video(cap, frame_number):
    # Read until video is completed
    cap.set(1, frame_number - 1)
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
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
    delta =




if __name__ == '__main__':
    vid = read_video(video_path)
    print(fps)
    display_video(vid, 445500)
    print(metadata)

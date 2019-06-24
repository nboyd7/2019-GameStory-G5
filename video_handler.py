# Given event find it in the video files
from datetime import timedelta

import cv2
import utils
import read_metadata as meta_util
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

START_TIME_DAY_1 = '2018-03-02T10:00:00.000+00:00'
START_TIME_DAY_2 = '2018-03-04T10:00:00.000+00:00'

global currentDigit


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
    cv2.waitKey(1000)
    ret, frame = cap.read()
    cv2.waitKey(1000)
    # Display the resulting frame
    shape = frame.shape
    roi = [0.007, 0.05, 0.435, 0.451]  # format ymin, ymax, xmin, xmax
    sub_frame = frame[round(roi[0] * shape[0]):round(roi[1] * shape[0]),
                round(roi[2] * shape[1]):round(roi[3] * shape[1]), :]
    ret, sub_frame = cv2.threshold(sub_frame, 127, 255, cv2.THRESH_BINARY)
    scale = 3
    # sub_frame_resized = cv2.resize(sub_frame, (round(sub_frame.shape[1] * scale), round(sub_frame.shape[0] * scale)))
    cv2.imshow('FrameScore', sub_frame)

    round_number_filename = f'digit_{currentDigit}.png'
    cv2.imwrite(round_number_filename, sub_frame)
    print(f'saved file {round_number_filename}')


    while (cap.isOpened()) and 0:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            # Display the resulting frame
            shape = frame.shape
            roi = [0.007, 0.05, 0.435, 0.451] #format ymin, ymax, xmin, xmax
            sub_frame = frame[round(roi[0]*shape[0]):round(roi[1]*shape[0]), round(roi[2]*shape[1]):round(roi[3]*shape[1]), :]
            ret, sub_frame = cv2.threshold(sub_frame, 127, 255, cv2.THRESH_BINARY)
            scale = 3
            #sub_frame_resized = cv2.resize(sub_frame, (round(sub_frame.shape[1] * scale), round(sub_frame.shape[0] * scale)))
            cv2.imshow('FrameScore', sub_frame)

            round_number_filename = f'digit_{currentDigit}.png'
            cv2.imwrite(round_number_filename, sub_frame)
            print(f'saved file {round_number_filename}')

            #cv2.imshow('Frame', frame)


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





if __name__ == '__main__':
    ss = meta_util.StreamSync()
    video_name = '2018-03-02_P1.mp4'
    video_path = "C:\save\\" + video_name
    vid = read_video(video_path)

    m1_p1_start_frame = int(ss.data['match_1']['player_P1'])
    print(m1_p1_start_frame)
    round_frame = []
    round_frame.append(m1_p1_start_frame + 5000)
    round_frame.append(m1_p1_start_frame + 18000)
    round_frame.append(m1_p1_start_frame + 20000)
    round_frame.append(m1_p1_start_frame + 25000)
    #round_test_frame = m1_p1_start_frame + 25000
    for i in range(0, len(round_frame)):
        currentDigit = i
        print(f'processing round number {i}')
        print(round_frame[i])
        display_video(vid, round_frame[i])
        vid = read_video(video_path)



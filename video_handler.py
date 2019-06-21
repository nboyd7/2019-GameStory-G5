# Given event find it in the video files

import cv2
import utils

START_TIME_DAY_1 = '2018-03-02T10:00:00.000+00:00'
START_TIME_DAY_2 = '2018-03-04T10:00:00.000+00:00'


# todo: make a dictionary that matches player names with the video file they are in P1 - P10
video_finder_day_1 = {
}

video_finder_day_2 = {
}


# return cv2 video capture object
def read_video(fp):
    vid = cv2.VideoCapture(fp)
    if (vid.isOpened() == False):
        print("Error opening video stream or file")
    return vid

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
def find_frame(event, start_date, vid):
    date_str = event['date']
    date = utils.convertDate(date_str)
    delta = date - start_date
    return calculate_frames(delta, vid.get(cv2.CAP_PROP_FPS))

def calculate_frames(date_delta, fps):
    seconds  = date_delta.total_seconds() + 70
    return fps * seconds

# get video file based on actor of event
def get_video_file(actor, day=1):
    if day == 1:
        return video_finder_day_1[actor]
    else:
        return video_finder_day_2[actor]


if __name__ == '__main__':
    video_path =
    vid = read_video(video_path)
    fps = vid.get(cv2.CAP_PROP_FPS)
    print(fps)
    display_video(vid, 445500)
    print(metadata)

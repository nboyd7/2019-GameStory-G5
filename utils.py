from datetime import datetime
import video_handler
import cv2

#master start frame is the start frame to which the json event log time is synced
master_start_frame = [443259]

#format: stream_starts[video_name][match_id] = match_round_1_start_frame
stream_starts = {
    '2018-03-02_P1': [443323],
    '2018-03-02_P2': [443323],
    '2018-03-02_P3': [443296],
    '2018-03-02_P4': [443292],
    '2018-03-02_P5': [443278],
    '2018-03-02_P6': [443241],
    '2018-03-02_P7': [443235],
    '2018-03-02_P8': [443220],
    '2018-03-02_P9': [443235],
    '2018-03-02_P10': [443200],
    '2018-03-02_P11': [438875]
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

def playerToPath(player_name):
    id = player_id[player_name]
    path = f'C:\save\\2018-03-02_P{id}.mp4'
    return path

def convertDate(indate):
    return datetime.strptime(indate[:-7] + '0000' , '%Y-%m-%dT%H:%M:%S.%f')


def second_to_frame_num(sec, fps):
    return sec * fps


# save frame to file when you quit out of video stream
# similar to video_handler display_video
def screenshot(fp, frame_number):
    # Read until video is completed
    cap = video_handler.read_video(fp)
    cap.set(1, frame_number - 1)
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        #display_score_frame(frame)
        if ret == True:
            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(5) & 0xFF == ord('q'):
                cv2.imwrite("frames/frame%d.jpg" % frame_number, frame)
                break

            frame_number = frame_number + 1

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


# Used for quick screen shot runs
if __name__ == '__main__':
    screenshot('/Volumes/Other 1/2018-03-02_P11.mp4', second_to_frame_num(11951, 60)
    return datetime.strptime(indate[:-7] + '0000', '%Y-%m-%dT%H:%M:%S.%f')

def event_time_to_stream_frame(dt, player_id, match_id):
    match_indx = match_id-1
    vid_stream_path = f'C:\save\\2018-03-02_P{player_id}.mp4'
    time_from_match_start = dt - event_log_match_start_time[match_indx]
    frame_from_match_start = round(time_from_match_start.seconds * 60 + (time_from_match_start.microseconds / 1000000) * 60)
    player_stream_frame_offset_from_master = stream_starts[f'2018-03-02_P{player_id}'][match_indx] - master_start_frame[match_indx]
    frame_from_player_stream_match_start = stream_starts[f'2018-03-02_P{player_id}'][match_indx] + player_stream_frame_offset_from_master + frame_from_match_start
    return frame_from_player_stream_match_start

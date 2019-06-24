from datetime import datetime
import video_handler
import cv2

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
    screenshot('/Volumes/Other 1/2018-03-02_P11.mp4', second_to_frame_num(11951, 60))





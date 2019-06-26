import numpy as np
import read_JSON
import utils as u
import video_handler
import cv2


if __name__ == '__main__':
    # library = np.load('features_1.npy')
    # query = np.load('query_features_1.npy')
    #
    # res = []
    # iter = 0
    # for frame in library:
    #     res.append(np.sum(np.abs(np.subtract(frame, query))))
    #     if res[iter] == 0:
    #         print(iter)
    #     iter += 1
    #
    # print(sorted(res))

    dH = read_JSON.DataHandler()
    ds = dH.eventWindowArrayPerActor

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

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = np.float32(gray)
                roi = [0.35, 0.65, 0.35,
                       0.65]
                size_1 = gray.shape[0]
                size_2 = gray.shape[1]

                gray = gray[round(roi[0] * size_1):round(roi[1] * size_1),
                       round(roi[2] * size_2):round(roi[3] * size_2)]
                dst = cv2.cornerHarris(gray, 4, 3, 0.04)
                harris_result = np.uint8(255 * dst / dst.max())
                np.save(f'./resources/event_queries/{actor}_w{window_count}_e{event_count}', harris_result)




from PIL import Image, ImageTk
import numpy as np
import experiment as ex
import feature_extractor as ft
import tkinter as tk
import video_handler as vidH
import utils as u
import read_JSON
from functools import partial
import cv2



class GUIController:
    def __init__(self):
        self.com_path = 'C:\save\\2018-03-02_P11.mp4'
        self.dH = read_JSON.DataHandler()
        self.guiDs = {}
        self.libraries = {'harris': np.load('./library_match_1_segment_harris.npy'),
                     'histogram': np.load('./library_match_1_segment_histogram.npy')}


    def setPlayer(self, player_name):
        self.guiDs = self.dH.eventWindowArrayPerActor[player_name]
        vid_path = u.playerToPath(player_name)
        top = tk.Toplevel()
        top.title(f'{player_name} event window selector')
        e = {}
        curr_row = 0
        w = tk.Label(top, text='Start Frame')
        w.grid(row=curr_row, column=0)
        w = tk.Label(top, text='End Frame')
        w.grid(row=curr_row, column=1)
        w = tk.Label(top, text='TimeInFrames')
        w.grid(row=curr_row, column=2)
        w = tk.Label(top, text='[ k | hs | pnt | hs+pnt ]')
        w.grid(row=curr_row, column=3)

        w = tk.Label(top, text='Matched')
        w.grid(row=curr_row, column=6)
        w = tk.Label(top, text='First event frame')
        w.grid(row=curr_row, column=7)

        for data in self.guiDs:
            matched = 'unknown'
            stored_window_match = u.getWindowStoredData(player_name, data['window'])
            if stored_window_match != None:
                matched = str(stored_window_match['matched'])

            curr_row += 1
            e_start_frame= u.event_time_to_stream_frame(data['start_time'],u.player_id[player_name], 1)
            e_end_frame = u.event_time_to_stream_frame(data['end_time'],u.player_id[player_name], 1)
            e_duration_frames = e_end_frame - e_start_frame
            e_stats = '[ ' + str(data['event_occurrences']['kill']) + ' | ' + str(data['event_occurrences']['kill_hs']) + ' | ' + str(data['event_occurrences']['kill_pnt']) + ' | ' + str(data['event_occurrences']['kill_hs_pnt']) + ' ]'

            w = tk.Label(top, text=e_start_frame)
            w.grid(row=curr_row, column=0)

            w = tk.Label(top, text=e_end_frame)
            w.grid(row=curr_row, column=1)

            w = tk.Label(top, text=e_duration_frames)
            w.grid(row=curr_row, column=2)

            w = tk.Label(top, text=e_stats)
            w.grid(row=curr_row, column=3)

            w = tk.Button(top, text='Play', width=10, bg='lightgreen', command=partial(vidH.display_video_extract, vid_path, e_start_frame, e_end_frame))
            w.grid(row=curr_row, column=4)

            w = tk.Button(top, text='Find Window', width=15, bg='purple',
                       command=partial(u.findWindow, self.libraries, data, player_name))
            w.grid(row=curr_row, column=5)

            w = tk.Label(top, text=matched)
            w.grid(row=curr_row, column=6)

            e_event_first_frame = u.event_time_to_stream_frame(data['event_times'][0],u.player_id[player_name], 1)
            w = tk.Label(top, text=e_event_first_frame)
            w.grid(row=curr_row, column=7)

            com_frame_number, nothing = u.findMatch(self.libraries, {}, player_name, data, 1)
            comm_frame_number_first = u.frameOffsetToCommentatorFrame(1, com_frame_number)
            com_frame_number, nothing = u.findMatch(self.libraries, {}, player_name, data, len(data['event_times']))
            comm_frame_number_last = u.frameOffsetToCommentatorFrame(1, com_frame_number)

            # dont end up showing way to long video segment
            if comm_frame_number_last < comm_frame_number_first:
                comm_frame_number_last = comm_frame_number_first
            elif comm_frame_number_last > comm_frame_number_first + 60*10:
                comm_frame_number_last = comm_frame_number_first + 60*10

            w = tk.Button(top, text='Play match', width=10, bg='lightgreen',
                          command=partial(vidH.display_video_extract, self.com_path, comm_frame_number_first-self.dH.getPadding()[0]*60, comm_frame_number_last+self.dH.getPadding()[1]*60))
            w.grid(row=curr_row, column=8)

            e_i = 0
            for event_time in data['event_times']:
                # w = tk.Button(top, text=f'Show e: {e_i+1}', width=10, bg='lightgreen',
                #            command=partial(vidH.read_frame, vid_path, u.event_time_to_stream_frame(event_time, u.player_id[player_name], 1)))
                # w.grid(row=curr_row, column=8+2*e_i)

                if stored_window_match != None:
                    # w = tk.Button(top, text=f'DispWinRes', width=10, bg='blue',
                    #               command=partial(self.displayWindowMatchResult, data, player_name, vid_path))
                    # w.grid(row=curr_row, column=9 + 2 * e_i)
                    w = tk.Button(top, text=f'Show Match: {e_i+1}', width=15, bg='lightblue',
                               command=partial(self.displayMatchResult, data, player_name, e_i+1, vid_path, u.event_time_to_stream_frame(event_time, u.player_id[player_name], 1)))
                    w.grid(row=curr_row, column=9 + 1 * e_i)
                else:
                    w = tk.Button(top, text=f'Find match: {e_i+1}', width=15, bg='blue',
                               command=partial(u.getMatch, self.libraries, data, player_name, e_i+1))
                    w.grid(row=curr_row, column=9+1*e_i)

                e_i += 1


        button = tk.Button(top, text='Close', width=25, command=top.destroy)
        button.grid(row=curr_row+1, column=0)
        top.mainloop()

    def displayWindowMatchResult(self, data, player_name, vid_path):
        event_times = data['event_times']
        event_number = 1
        render_player = []
        render_comm = []
        for e_t in event_times:
            player_stream_frame = u.event_time_to_stream_frame(e_t, u.player_id[player_name], 1)
            player_frame = vidH.get_frame(vid_path, player_stream_frame)
            com_frame, nothing = u.findMatch(self.libraries, {}, player_name, data, event_number)
            comm_frame = vidH.get_frame(self.com_path, u.frameOffsetToCommentatorFrame(1, com_frame))
            render_player.append(ImageTk.PhotoImage(Image.fromarray(player_frame)))
            render_comm.append(ImageTk.PhotoImage(Image.fromarray(comm_frame)))
            event_number += 1
        h = player_frame.shape[0]
        w = player_frame.shape[1]
        top = tk.Toplevel()
        canvas = tk.Canvas(top, width=w * 2, height=h * len(event_times))
        canvas.pack(expand=True)
        top.title('Match Results')
        image_player = []
        image_comm = []
        # for i in range(0, len(render_player)):
        canvas.create_image((w / 2, h / 2 + (event_number - 1) * h), image=render_player[0])
        canvas.create_image((w + w / 2, h / 2 + (event_number - 1) * h), image=render_comm[0])
        tk.mainloop()

    def displayMatchResult(self, data, player_name, event_number, vid_path, player_stream_frame):
        player_frame = cv2.cvtColor(vidH.get_frame(vid_path, player_stream_frame), cv2.COLOR_BGR2RGB)
        com_frame, nothing = u.findMatch(self.libraries, {}, player_name, data, event_number)
        comm_frame = cv2.cvtColor(vidH.get_frame(self.com_path, u.frameOffsetToCommentatorFrame(1, com_frame)), cv2.COLOR_BGR2RGB)
        render_player = ImageTk.PhotoImage(Image.fromarray(player_frame))
        render_comm = ImageTk.PhotoImage(Image.fromarray(comm_frame))
        h = player_frame.shape[0]
        w = player_frame.shape[1]
        top = tk.Toplevel()
        canvas = tk.Canvas(top, width=w*2, height=h)
        canvas.pack(fill=tk.BOTH, expand=True)
        top.title('Match Results')
        canvas.create_image((w/2, h/2), image=render_player)
        h = comm_frame.shape[0]
        w = comm_frame.shape[1]
        image_comm = canvas.create_image((w+w/2, h/2), image=render_comm)
        tk.mainloop()

    def openSettings(self):
        top = tk.Toplevel()
        top.title('Settings')

        curr_row = 0

        self.addSetting(top, 'padding_before:', self.dH.getPadding()[0], curr_row, self.dH.setStartPadding)
        curr_row += 1
        self.addSetting(top, 'padding_after:', self.dH.getPadding()[1], curr_row, self.dH.setEndPadding)
        curr_row += 1
        self.addSetting(top, 'window_size:', self.dH.getWindowSize(), curr_row, self.dH.setWindowSize)
        curr_row += 1
        self.addSetting(top, 'match_thresh:', 0, curr_row, u.updateWindowMatches)
        curr_row += 1
        # self.addSetting(top, 'window_straddle:', self.dH.getWindowStraddle(), curr_row, self.dH.setWindowStraddle)
        # curr_row += 1

        button = tk.Button(top, text='Close', width=25, command=top.destroy)
        button.grid(row=curr_row + 1, column=0)
        top.mainloop()

    def addSetting(self, master, desc, val, curr_row, ret_func):
        w = tk.Label(master, text=f'{desc}')
        w.grid(row=curr_row, column=0)

        e = tk.Entry(master)
        e.grid(row=curr_row, column=1)
        e.delete(0, tk.END)
        e.insert(0, f"{val}")

        button = tk.Button(master, text='Update', width=18, command=partial(self.updateSetting, ret_func, e))
        button.grid(row=curr_row, column=2)

    def updateSetting(self, ret_func, entry):
        ret_func(entry.get())

    def openRankedList(self):
        top = tk.Toplevel()
        top.title('Ranked List')

        rl = u.getRankedList()

        for row in rl:
            weight = row['weight']
            name = row['window_name']
            w = tk.Label(top, text=f'{weight} {name}')
            w.pack()

        top.mainloop()

if __name__ == '__main__':
    controller = GUIController()


    r = tk.Tk()

    r.title('GUI for event window navigation')

    w=tk.Text(r, height = 2, width = 37)
    w.pack()
    w.insert(tk.END, 'Select player to view event windows:')
    for player,stream in vidH.player_id.items():
        w = tk.Button(r, text=f'{player}', command=partial(controller.setPlayer,player))
        w.pack()

    button = tk.Button(r, text='Ranked List', width=25, command=controller.openRankedList)
    button.pack()
    button = tk.Button(r, text='Settings', width=25, command=controller.openSettings)
    button.pack()
    button = tk.Button(r, text='Stop', width=25, command=r.destroy)
    button.pack()
    r.mainloop()

from tkinter import *
import video_handler as vidH
import utils as u
import read_JSON
from functools import partial



class GUIController:
    def __init__(self):
        self.dH = read_JSON.DataHandler()
        self.guiDs = {}


    def setPlayer(self, player_name):
        self.guiDs = self.dH.eventWindowArrayPerActor[player_name]
        vid_path = u.playerToPath(player_name)
        top = Toplevel()
        top.title(f'{player_name} event window selector')
        e = {}
        curr_row = 0
        w = Label(top, text='Start Frame')
        w.grid(row=curr_row, column=0)
        w = Label(top, text='End Frame')
        w.grid(row=curr_row, column=1)
        w = Label(top, text='TimeInFrames')
        w.grid(row=curr_row, column=2)
        w = Label(top, text='[ k | hs | pnt | hs+pnt ]')
        w.grid(row=curr_row, column=3)

        for data in self.guiDs:
            curr_row += 1
            e_start_frame= u.event_time_to_stream_frame(data['start_time'],u.player_id[player_name], 1)
            e_end_frame = u.event_time_to_stream_frame(data['end_time'],u.player_id[player_name], 1)
            e_duration_frames = e_end_frame - e_start_frame
            e_stats = '[ ' + str(data['event_occurrences']['kill']) + ' | ' + str(data['event_occurrences']['kill_hs']) + ' | ' + str(data['event_occurrences']['kill_pnt']) + ' | ' + str(data['event_occurrences']['kill_hs_pnt']) + ' ]'

            w = Label(top, text=e_start_frame)
            w.grid(row=curr_row, column=0)

            w = Label(top, text=e_end_frame)
            w.grid(row=curr_row, column=1)

            w = Label(top, text=e_duration_frames)
            w.grid(row=curr_row, column=2)

            w = Label(top, text=e_stats)
            w.grid(row=curr_row, column=3)

            w = Button(top, text='Play', width=15, bg='lightgreen', command=partial(vidH.display_video_extract, vid_path, e_start_frame, e_end_frame))
            w.grid(row=curr_row, column=4)

        button = Button(top, text='Close', width=25, command=top.destroy)
        button.grid(row=curr_row+1, column=0)
        top.mainloop()

    def openSettings(self):
        top = Toplevel()
        top.title('Settings')

        curr_row = 0

        self.addSetting(top, 'padding_before:', self.dH.getPadding()[0], curr_row, self.dH.setStartPadding)
        curr_row += 1
        self.addSetting(top, 'padding_after:', self.dH.getPadding()[1], curr_row, self.dH.setEndPadding)
        curr_row += 1
        self.addSetting(top, 'window_size:', self.dH.getWindowSize(), curr_row, self.dH.setWindowSize)
        curr_row += 1
        # self.addSetting(top, 'window_straddle:', self.dH.getWindowStraddle(), curr_row, self.dH.setWindowStraddle)
        # curr_row += 1

        button = Button(top, text='Close', width=25, command=top.destroy)
        button.grid(row=curr_row + 1, column=0)
        top.mainloop()

    def addSetting(self, master, desc, val, curr_row, ret_func):
        w = Label(master, text=f'{desc}')
        w.grid(row=curr_row, column=0)

        e = Entry(master)
        e.grid(row=curr_row, column=1)
        e.delete(0, END)
        e.insert(0, f"{val}")

        button = Button(master, text='Update', width=18, command=partial(self.updateSetting, ret_func, e))
        button.grid(row=curr_row, column=2)

    def updateSetting(self, ret_func, entry):
        ret_func(entry.get())


if __name__ == '__main__':
    controller = GUIController()

    r = Tk()
    r.title('GUI for event window navigation')

    w=Text(r, height = 2, width = 37)
    w.pack()
    w.insert(END, 'Select player to view event windows:')
    for player,stream in vidH.player_id.items():
        w = Button(r, text=f'{player}', command=partial(controller.setPlayer,player))
        w.pack()

    button = Button(r, text='Settings', width=25, command=controller.openSettings)
    button.pack()
    button = Button(r, text='Stop', width=25, command=r.destroy)
    button.pack()
    r.mainloop()
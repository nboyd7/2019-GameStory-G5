import os
import csv
import re


class StreamSync:

    def __init__(self):
        self.data = {}
        self.csv_folder = 'resources/sync/'
        self.MatchCount = 12 #number of total matches
        self.LoadAllMatches()

    def LoadAllMatches(self):
        for match_id in range(1,self.MatchCount):
            if match_id == 10: #match with id 10 has no sync file
                continue
            self.LoadMatchSync(match_id)


    def GetSyncMatchFilename(self, match_id, stream_id):
        basename_start = 'sync_match_'
        basename_mid = '_P'
        extension = '.csv'
        return basename_start + str(match_id) + basename_mid + str(stream_id) + extension


    def LoadMatchSync(self, match_id):
        filenames = []
        match_key = f'match_{match_id}'
        self.data[match_key] = {}
        for i in range(1, 12): #load all player streams
            player_key = f'player_P{i}'
            filenames.append(self.GetSyncMatchFilename(match_id, i))
            start_frame = self.GetStartFrame(filenames[i-1])
            self.data[match_key][player_key] = start_frame

    def GetStartFrame(self, csv_path, round=1, loud=False):
        with open(self.csv_folder + csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    if loud:
                        print(f'Column names are {", ".join(row)}')
                    line_count += 1
                elif line_count == round:
                    if loud:
                        print(f'\t match: {row[0]} round: {row[1]} round_begin: {row[2]} frame_nr: {row[3]}.')
                    start_frame = row[3]
                    line_count += 1
                    break
                else:
                    line_count += 1
            if loud:
                print(f'Processed {line_count} lines.')
            return start_frame


if __name__ == '__main__':
    ss = StreamSync()
    for match in ss.data:
        print(match)
        for player in ss.data[match]:
            print(f'player: {player} start_frame: {ss.data[match][player]}')

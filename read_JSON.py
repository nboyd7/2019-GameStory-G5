# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:50:40 2019

@author: olivi
"""
"""
MMSR final
JSON to dict,
extract events,
make eventWeight dict,
function eventTimeLineToWindowArray(winSize, )
"""

from datetime import datetime
from datetime import timedelta
import copy
import json
import utils
import math


class DataHandler:

    def __init__(self):
        self.JSON_relative_path = 'resources/1.json'

        #TODO try different window sizes
        self.windowSize = 8;
        self.windowStraddle = self.windowSize/2;

        #TODO limit min and max date to start of match and end of match (and for commentator stream till end of highlights
        self.minDate = datetime.now()
        self.maxDate = datetime.now()

        self.actors = []
        self.eventWindowArrayPerActor = {}
        self.eventIdsPerWindowPerActor = {}

        # event Type Weight dictionary
        self.eventTypeWeight = {
            "kill": 1,
            "kill_hs": 2,
            "kill_pnt": 1.5,
            "kill_hs_pnt": 3
        }
        self.dataset = []
        self.loadFile()
        self.assertWeightPerEvent()
        self.createWindowTimeLine()
        self.cleanWindowTimeLine()



    #only saves the desired events
    def json2dataset(self, jsonFile):
        result_array = []
        #init min and max dates
        self.minDate = utils.convertDate(jsonFile[0]["date"])
        self.maxDate = utils.convertDate(jsonFile[0]["date"])
        for i in range(0, len(jsonFile)-1):
            result_dict = {}
            jobj = jsonFile[i]
            jdata = jobj["data"]
            jobj_type = jobj["type"]
            if jobj_type == 'kill':
                event_type = 'kill'
                event_actor = jdata["actor"]["playerId"]
                if event_actor not in self.actors:
                    self.actors.append(event_actor)
                event_victim = jdata["victim"]["playerId"]
                if jdata["headshot"] == True:
                    event_type = event_type + '_hs'
                if jdata["penetrated"] == True:
                    event_type = event_type + '_pnt'
                result_dict["event_type"] = event_type
                result_dict["date"] = utils.convertDate(jobj["date"]) #use python datetime format for dates
                #update min and max dates of session
                if result_dict["date"] < self.minDate:
                    self.minDate = result_dict["date"]
                elif result_dict["date"] > self.maxDate:
                    self.maxDate = result_dict["date"]
                result_dict["actor"] = event_actor
                result_dict["victim"] = event_victim
                result_dict["event_id"] = i
                result_array.append(result_dict)
        self.dataset = result_array

    #loads JSON file and converts to dataset
    def loadFile(self):
        with open(self.JSON_relative_path, 'r') as f:
            datastore = json.load(f)
            self.json2dataset(datastore)

    #check if every dataset event has a corresponding weight in the eventTypeWeight dict
    def assertWeightPerEvent(self):
        eventTypes = []
        for indx in range(0, len(self.dataset)):
            tempEventType = self.dataset[indx]['event_type']
            if tempEventType not in eventTypes:
                eventTypes.append(tempEventType)
        for eT in eventTypes:
            try:
                assert eT in self.eventTypeWeight, f"Error not all events have weights \'{eT}\'"
            except AssertionError as e:
                print(e)

    #uses the event timeline dataset and window size to create a window array of windowData dictionaries with the data of what happens in that time window
    def createWindowTimeLine(self):
        """use window size and straddle to slide a window over the event timeline dataset to create a windowData dict array"""
        w_size = self.windowSize
        w_strad = self.windowStraddle
        windowDataDict = {
            "start_time": 0,
            "end_time": 0,
            "event_occurrences" : dict.fromkeys(self.eventTypeWeight.keys(), 0),
            "total_weight": 0,
        }
        time_delta = self.maxDate - self.minDate
        window_count = math.ceil(time_delta.seconds / w_size)

        actorWindows = dict.fromkeys(self.actors, [])
        actorEventIdsPerWindow = dict.fromkeys(self.actors, [])
        for actor in self.actors: #window array of windowData dicts per actor
            actorWindow = []
            eventIdsPerWindow = []
            for iter in range(0, window_count):
                w_dict = copy.deepcopy(windowDataDict)
                w_dict["start_time"] = self.minDate + timedelta(seconds=w_strad * iter)
                w_dict["end_time"] = w_dict["start_time"] + timedelta(seconds=w_size)

                eventIdsInWindow = []
                for event in self.dataset:
                    if event["actor"] == actor and w_dict["start_time"] < event["date"] < w_dict["end_time"]: #get events in window
                        w_dict["event_occurrences"][event["event_type"]] += 1
                        w_dict["total_weight"] += self.eventTypeWeight[event["event_type"]]
                        eventIdsInWindow.append(event["event_id"])
                actorWindow.append(w_dict)
                eventIdsPerWindow.append(eventIdsInWindow)
            actorWindows[actor] = actorWindow
            actorEventIdsPerWindow[actor] = eventIdsPerWindow
        self.eventWindowArrayPerActor = actorWindows
        self.eventIdsPerWindowPerActor = actorEventIdsPerWindow

    def cleanWindowTimeLine(self):
        """events can occur several times in overlapping windows, the idea is to remove redundant windows, of which all events are also present in another window. Then keep the window with the highers weight or if equivalent weights then keep the earlier window (to see the action leading up to the events)"""
        windowTl = self.eventWindowArrayPerActor
        eventIds = self.eventIdsPerWindowPerActor
        for actor in self.actors:
            indx_to_del = []
            for iter in range(0, len(windowTl[actor])-2):
                next_iter = iter + 1
                e_curr = eventIds[actor][iter]
                e_next = eventIds[actor][next_iter]
                if set(e_next).issubset(set(e_curr)): #if all events in next window are included in current window then discard next window
                    indx_to_del.append(next_iter)
                    if next_iter not in indx_to_del:
                        indx_to_del.append(next_iter)
                elif set(e_curr).issubset(set(e_next)):#set(e_curr) has less events than set(e_next) and is a subset of the latter
                    if iter not in indx_to_del:
                        indx_to_del.append(iter)
            for indx in sorted(indx_to_del, reverse=True):
                del eventIds[actor][indx]
                del windowTl[actor][indx]
        self.eventWindowArrayPerActor = windowTl
        self.eventIdsPerWindowPerActor = eventIds



def json2dataset(jsonFile):
    result_array = []
    for i in range(0, len(jsonFile) - 1):
        result_dict = {}
        jobj = jsonFile[i]
        jdata = jobj["data"]
        jobj_type = jobj["type"]
        if jobj_type == 'kill':
            event_type = 'kill'
            event_actor = jdata["actor"]["playerId"]
            event_victim = jdata["victim"]["playerId"]
            if jdata["headshot"] == True:
                event_type = event_type + '_hs'
            if jdata["penetrated"] == False:
                event_type = event_type + '_pnt'
            result_dict["event_type"] = event_type
            result_dict["date"] = jobj["date"]
            result_dict["event_actor"] = event_actor
            result_dict["event_victim"] = event_victim
            result_array.append(result_dict)

    return result_array


# todo: go over our dataset and extract eventType to be weighed
def weighEventType(dataset):
    for event in dataset:
        if event["event_type"] == "kill":
            event["weight"] = 1
        if event["event_type"] == "kill_hs":
            event["weight"] = 2
        if event["event_type"] == "kill_pnt":
            event["weight"] = 2
        if event["event_type"] == "kill_hs_pnt":
            event["weight"] = 3
    return dataset



# takes eventType list and converts to eventTypeWeight dictionary
# def createEventTypeWeights(events):


if __name__ == '__main__':

    filename = 'resources/1.json'
    dataset = {}
    with open(filename, 'r') as f:
        datastore = json.load(f)
        dataset = json2dataset(datastore)

        weighted_dataset = weighEventType(dataset)
        sorted_weighted_dataset = sorted(weighted_dataset, key=lambda k: k["weight"], reverse=True)
        print(sorted_weighted_dataset)

    myDl = DataHandler()

    for actor in myDl.actors:
        print(actor)
        for i in range(0, len(myDl.eventWindowArrayPerActor[actor])):
            if myDl.eventWindowArrayPerActor[actor][i]["total_weight"] > 0:
                print(myDl.eventWindowArrayPerActor[actor][i])

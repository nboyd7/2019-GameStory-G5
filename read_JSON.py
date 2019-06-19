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

import json


def json2dataset(jsonFile):
    result_array = []
    for i in range(0, len(jsonFile)-1):
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

#todo: go over our dataset and extract eventType to be weighed
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


#takes eventType list and converts to eventTypeWeight dictionary
#def createEventTypeWeights(events):


if __name__ == '__main__':
    filename = 'resources/1.json'
    dataset = {}
    with open(filename, 'r') as f:
        datastore = json.load(f)
        dataset = json2dataset(datastore)

	weighted_dataset = weighEventType(dataset)
	print(weighted_dataset)

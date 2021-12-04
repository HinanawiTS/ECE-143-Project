import pandas as pd
import json
import numpy as np

import os

def read_preprocess_events(directory): 
    """ 
    Read and preprocess event datasets in the given directory into one Pandas DataFrame. 
    Preprocess includes dropping unnecessary columns and add new features columns. 
    
    :param directory: directory of datasets
    :type directory: string 
    :returns: preprocessed dataset
    
    """ 
    
    assert isinstance(directory, str)
    assert len(directory) > 0
    
    # Read dataset
    england = pd.read_json(directory + "/events_England.json")
    england = england[england["subEventName"] == "Shot"]
    for i in os.listdir(directory): 
        if i != "events_England.json": 
            dataset = pd.read_json(directory + "/" + i)
            england = pd.concat([england, dataset], ignore_index = True) 
    
    # Process dataset  
    england = england[england["subEventName"] == "Shot"]
    eng = england.drop(["eventId", "eventName", "subEventId", "subEventName"], axis = 1)
    eng["taglist"] = eng.apply(lambda x: [i["id"] for i in x["tags"]], axis = 1)
    eng = eng.drop("tags", axis = 1) 

    eng = eng.rename({"playerId": "wyId"}, axis = 1)
   
    # Adding features
    eng[["Goal", "Header", "Counter Attack", "Right Foot", "Blocked", "First Half", "X", "Y", "C", "Distance", "angle_degrees"]] = 0 
    
    return eng

def read_process_players(directory): 
    """ 
    Read and process the players dataset in the given directory. 
    Preprocess includes dropping unnecessary columns and add new features columns. 
    
    :param directory: directory of dataset 
    :type directory: string 
    :returns: processed players dataset 
    
    :type returns: Pandas DataFrame 
    
    """  
    
    assert isinstance(directory, str) 
    assert len(directory) > 0
    
    # Read dataset
    players = pd.read_json(directory + "/players.json")
    
    # Preprocess players
    players["country"] = players.apply(lambda x: x["passportArea"]["name"], axis = 1)
    players["rol"] = players.apply(lambda x: x["role"]["name"], axis = 1)
    
    # Drop unnecessary columns 
    players = players.drop(["passportArea", "role", "birthArea"], axis = 1) 
    
    return players 

def process_row(row): 
    """ 
    Process given row, create new features given a list of tags in the "taglist" column. If the tag 
    appears in the taglist the feature corresponding to the tag is set to 1. Added features are: 
    Header: control or shoot balls using head, corresponding to tag 403. 
    Goal: given shoot was a goal, corresponding to tag 101. 
    Counter Attack: a counter attack, corresponding to tag 1901. 
    Blocked: shoot was blocked, corresponding to tag 2101. 
    Right Foot: using right foot, corresponding to tag 402. 
    First Half: was in first half of the match, corresponding to "matchPeriod" == 1. 
    X: X coordinate of the shoot. 
    Y: Y coordinate of the shoot. 
    C: distance in Y direction from the goal. 
    Distance: distance of the shoot. 
    angle_degrees: angle of the shoot converted to degrees. 
    
    
    
    :param row: given row of pandas DataFrame
    :type row: Pandas Series
    :returns: given row with new features
    
    :type returns: Pandas Series
    
    """ 
    
    assert isinstance(row, pd.Series) 
    assert len(row) > 0
    
    taglist = row["taglist"]
    
    if 403 in taglist: 
        row["Header"] = 1
        
    if 101 in taglist: 
        row["Goal"] = 1
        
    if 1901 in taglist: 
        row["Counter Attack"] = 1 
        
    if 2101 in taglist: 
        row["Blocked"] = 1 
    
    if 402 in taglist: 
        row["Right Foot"] = 1 
    
    if row['matchPeriod'] == "1H":
        row['First Half'] = 1

    row['X'] = 100 - row['positions'][0]['x']
    row['Y'] = row['positions'][0]['y']
    row['C'] = abs(row['positions'][0]['y'] - 50)
    
    x = row['X'] * 105 / 100
    y = row['C'] * 65 / 100
    row['Distance'] = np.sqrt(x ** 2 + y ** 2)
        
    angle = np.arctan(7.32 * x / (x ** 2 + y ** 2 - (7.32 / 2) ** 2))
        
    if angle < 0:
        angle = np.pi + angle
        
    row['angle_degrees'] = angle * 180 / np.pi
    
    return row 

def strong(row): 
    """ 
    Given shoot was executed by the player's strong foot or not.  
    
    :param row: given row
    :type row: pandas series
    :returns: is strong foot or not 
    
    :type returns: boolean integer
    
    """ 
    
    assert isinstance(row, pd.Series)
    
    if ((row["Right Foot"] == 1) and (row["foot"] == "right")): 
        return 1
    elif ((row["Right Foot"] == 0) and (row["foot"] == "left")): 
        return 1
    else: 
        return 0 

def adding_features_combine(event, players): 
    """ 
    Adding and modifying features using the process_row and strong function, combining 
    events dataset with the players dataset
    
    :param event: the preprocessed dataset of events
    :type eventt: Pandas DataFrame 
    :param players: the preprocessed dataset of players
    
    :type players: Pandas DataFrame 
    :returns: dataset with new features added or modified 
    :type returns: Pandas DataFrame
    
    """  
    
    assert isinstance(event, pd.DataFrame)
    assert isinstance(players, pd.DataFrame)
    assert len(event) > 0
    
    # Adding features using process_row function
    dataset = event.apply(process_row, axis = 1)
    dataset = dataset.drop(["positions", "matchPeriod", "taglist"], axis = 1) 
    
    # Merging event and players dataset
    dataset = dataset.merge(players, on = "wyId")
    
    # Adding features using function strong_foot 
    dataset["strong foot"] = dataset.apply(strong, axis = 1) 

    return dataset 

def process_dataset(directory): 
    """ 
    Read and process events and players datasets in the given directory, 
    adding features using all functions above. 
    
    :param directory: directory of datasets 
    :type directory: string
    
    :returns: processed dataset with all events and players 
    :type returns: Pandas DataFrame 
    
    """  
    
    assert isinstance(directory, str) 
    assert len(directory) > 0
    
    # Read and preprocess events and players dataset 
    event = read_preprocess_events(directory + "/events") 
    players = read_process_players(directory) 
    
    # Adding features 
    dataset = adding_features_combine(event, players) 
    
    return dataset 
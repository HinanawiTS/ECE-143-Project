#!/usr/bin/env python
# coding: utf-8

import json
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def shot_matrix(eventdata):
    assert isinstance(eventdata, str)
    with open(eventdata) as f:
        data = json.load(f)
    
    #lets create the dataframe that we want to store our data in and all the attributes we are interested in 
    shots_dataset = pd.DataFrame(columns=['Goal','x','y','playerid','teamid','matchid','header'])
    
    #remember that the jsonfiles include passes, shots, tackles etc so we need to filter through these
    #lets find all the occurences of a shot within the set
    #refer to link in the prevous cell for info on the Wyscout event dataset, including tag names
    event_df = pd.DataFrame(data)
    all_shots = event_df[event_df['subEventName']=='Shot']
    
    #now we need to fill in our shots_dataset matrix by attribute columns
    #we will do this by filtering through the all-shot df (dataframe) we just made
    for index,shot in all_shots.iterrows():
        #here we fill in the columns for goals and headers with binary descripters
        shots_dataset.at[index,'Goal']=0
        shots_dataset.at[index,'header']=0
        for tag in shot['tags']:
            if tag['id']==101:
                shots_dataset.at[index,'Goal']=1
            elif tag['id']==403:
                shots_dataset.at[index,'header']=1
                
        #now we are interested in distance from the goal as well as the angle formed with the goal
        #Wyscouts pitch has its origin at the top left of the pitch and is 100m x 100m
        #therefore x and y represent percentage of nearness to top left corner 
        #most pitches are 105 meters by 68 so we will go with that
        shots_dataset.at[index,'Y']=shot['positions'][0]['y']*.68
        shots_dataset.at[index,'X']= (100 - shot['positions'][0]['x'])*1.05
        
        #now we use dummy variables x and y to calc distance and angle attributes
        shots_dataset.at[index,'x']= 100 - shot['positions'][0]['x'] 
        shots_dataset.at[index,'y']=shot['positions'][0]['y']
        shots_dataset.at[index,'Center_dis']=abs(shot['positions'][0]['y']-50)
        
        x = shots_dataset.at[index,'x']*1.05
        y = shots_dataset.at[index,'Center_dis']*.68
        shots_dataset.at[index,'Distance'] = np.sqrt(x**2 + y**2)
        
        #we are interested in the angle made between the width of the goal and the
        #straight line distance to the shot location. A goal is 7.32 meters wide
        #use the law of cosines
        c=7.32
        a=np.sqrt((y-7.32/2)**2 + x**2)
        b=np.sqrt((y+7.32/2)**2 + x**2)
        k = (c**2-a**2-b**2)/(-2*a*b)
        gamma = np.arccos(k)
        if gamma<0:
            gamma = np.pi + gamma
        shots_dataset.at[index,'Angle Radians'] = gamma
        shots_dataset.at[index,'Angle Degrees'] = gamma*180/np.pi
        
        #lastly we add the identifiers for player, team and match
        shots_dataset.at[index,'playerid']=shot['playerId']
        shots_dataset.at[index,'matchid']=shot['matchId']
        shots_dataset.at[index,'teamid']=shot['teamId']
        
        
    return shots_dataset

def get_tiddy_data(df):
    """
    Get tiddy dataset. Data cleaning.
    """
    assert isinstance(df, pd.DataFrame)

    #find out if the error is producing nan values
    df.isnull().values.any()

    #find how many such nan values
    df.isnull().sum().sum()

    df[df.isnull().any(axis=1)]

    df.dropna()

    df.drop(columns = ['x','y','Center_dis'])

    return df

def plot_violin_goal_dist_angle(df):
    # AssertionError
    assert isinstance(df, pd.DataFrame)
    #use the seaborn library to inspect the distribution of the shots by result (goal or no goal) 
    fig, axes = plt.subplots(1, 2,figsize=(11, 5))

    #use seaborn lib for violin plot and extract necessary columns from our dataframe df
    shot_dist = sns.violinplot(x="Goal", y="Distance",
                               data=df, inner="quart",ax= axes[0])
    shot_dist.set_xlabel("Goal? (0=no, 1=yes)", fontsize=12)
    shot_dist.set_ylabel("Distance (m)", fontsize=12)
    shot_dist.set_title("Distance of Shot from Goal vs. Result", fontsize=17, weight = "bold")
    shot_dist.set_ylim([0,45])

    #similar as before
    shot_ang = sns.violinplot(x="Goal", y="Angle Degrees",
                              data=df, inner="quart",ax = axes[1])
    shot_ang.set_xlabel("Goal? (0=no, 1=yes)", fontsize=12)
    shot_ang.set_ylabel("Angle (Degrees)", fontsize=12)
    shot_ang.set_title("Shot angle vs. Result", fontsize=17, weight = "bold");

def plot_violin_header_dist_angle(df):
    # AssertionError
    assert isinstance(df, pd.DataFrame)
    fig, axes = plt.subplots(1, 2,figsize=(11, 5))
    #the hue parameter splits plot by categorical data, in this case headers
    shot_distance = sns.violinplot(x="Goal", y="Distance",hue='header',
                                   data=df, inner="quart",split=True,ax = axes[0])
    shot_distance.set_xlabel("Goal? (0=no, 1=yes)", fontsize=12)
    shot_distance.set_ylabel("Distance (m)", fontsize=12)
    shot_distance.set_title("Distance of Shot from Goal vs. Result", fontsize=17, weight = "bold")
    shot_distance.set_ylim([0,45])
    shot_ang = sns.violinplot(x="Goal", y="Angle Degrees",hue='header',
                              data=df, inner="quart",split=True,ax = axes[1])
    shot_ang.set_xlabel("Goal? (0=no, 1=yes)", fontsize=12)
    shot_ang.set_ylabel("Angle (Degrees)", fontsize=12)
    shot_ang.set_title("Shot angle vs. Result", fontsize=17, weight = "bold");

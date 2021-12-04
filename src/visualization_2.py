#!/usr/bin/env python
# coding: utf-8

# # Data Loader
# #### Extracting data and adjusting data to numerical ones. 
# #### Adding distance and angle by calculating with (x, y) point.
# 

# In[4]:



from matplotlib import pyplot as plt
import seaborn as sns
import os

def correlation(shots_model):
    plt.figure(figsize=(16,12))
    ax = plt.axes()
    sns.heatmap(shots_model.corr(),annot=True,cmap='viridis')
    ax.set_title("Features Correlation Map")
    #plt.savefig('./image/CorrelationMap_AllEvent.jpg')
    plt.show()


# # Analysis of Distance and Angle by Goal
# ### this analysis can help user realized the relation among distance, angle and goal

# In[17]:

def analysisAD(shots_model):
    plt.figure(figsize=(10,6))
    ax = plt.axes()
    sns.scatterplot(x = 'Distance', y = 'angle_degrees', hue = 'Goal',size="Goal", sizes=(60, 30), 
                    palette='dark', data = shots_model, style="Goal", markers = {0:"^", 1:"o"})
    plt.xlabel("Distance (meters)")
    plt.ylabel("Angle (degrees)")
    ax.set_title("Analysis of Distance and Angle by Goal")
    #plt.savefig('./image/DistanceAngle_Analysis.jpg')
    plt.show()


# # Top View Image of Position and Goal
# ### the top view image is a good way to show the distribution of each shooting spot

# In[18]:

def topview(shots_model):
    plt.figure(figsize=(10,6))
    ax = plt.axes()
    sns.scatterplot(x = 'X', y = 'Y', hue = 'Goal',size="Goal", sizes=(30, 15), 
                    palette='dark', data = shots_model, style="Goal", markers = {0:"X", 1:"o"})
    plt.xlabel("X (meters)")
    plt.ylabel("Y (degrees)")
    ax.set_title("Top View Image of Position and Goal")
    #plt.savefig('./image/Position_TopView.jpg')
    plt.show()


# In[ ]:




